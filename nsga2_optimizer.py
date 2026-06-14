"""
NSGA-II Optimizer for Mask-Aware Steganography

This module implements NSGA-II to optimize:
- Block-based embedding mask
- Alpha (embedding strength)
- Optional payload ratio / embedding strength

The optimizer works with a pre-trained/frozen CNN encoder-decoder.

Each individual contains:
- mask_genes: Block-level binary genes [block_h, block_w]
- alpha: Embedding strength scalar
- optional payload_ratio: Embedding density

Objectives:
- f1 = -(normalized_psnr + normalized_ssim)  # Imperceptibility (minimize)
- f2 = -bpp  # Payload (minimize, higher is better so we minimize negative)
- f3 = stego_detection_probability  # Security (minimize)

Constraints:
- BER threshold: if BER > threshold, apply penalty
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import random
import copy
import logging

from metrics import (
    calculate_psnr, calculate_ssim, calculate_bpp, calculate_ber,
    calculate_security_score, create_block_mask, upsample_block_mask
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class NSGA2Config:
    """Configuration for NSGA-II optimization"""
    # Population parameters
    population_size: int = 30
    n_generations: int = 50

    # Evolutionary operator rates
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1

    # Image parameters
    image_height: int = 128
    image_width: int = 128
    message_length: int = 30

    # Block-based mask parameters
    block_size: int = 8  # Block size for mask (mask_chromosome_size = H//block_size x W//block_size)

    # Mask parameters
    initial_density: float = 0.3  # Initial fraction of embedding blocks
    min_density: float = 0.1
    max_density: float = 0.8

    # Alpha parameters
    initial_alpha: float = 1.0
    min_alpha: float = 0.1
    max_alpha: float = 2.0

    # Objective normalization
    psnr_weight: float = 1.0
    ssim_weight: float = 1.0

    # Constraint
    ber_threshold: float = 0.1  # Max acceptable BER

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def block_grid_h(self) -> int:
        return self.image_height // self.block_size

    @property
    def block_grid_w(self) -> int:
        return self.image_width // self.block_size


# ==============================================================================
# INDIVIDUAL / CHROMOSOME
# ==============================================================================

@dataclass
class Individual:
    """
    Represents an individual in NSGA-II population.

    Chromosome:
    - mask_genes: Binary genes for block mask [block_grid_h, block_grid_w]
    - alpha: Embedding strength scalar
    """
    mask_genes: np.ndarray  # Binary mask genes
    alpha: float  # Embedding strength

    fitness: Optional[np.ndarray] = None  # [imperceptibility, payload, security]
    rank: Optional[int] = None
    crowding_distance: Optional[float] = None

    def __post_init__(self):
        """Validate individual after initialization"""
        self.mask_genes = np.clip(self.mask_genes, 0, 1).astype(np.uint8)
        self.alpha = float(np.clip(self.alpha, 0.1, 2.0))

    def to_mask_tensor(self, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
        """Convert genes to full resolution mask tensor"""
        mask = upsample_block_mask(
            torch.from_numpy(self.mask_genes).unsqueeze(0),
            target_h, target_w
        )
        return mask.to(device)

    def to_alpha_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert alpha to tensor"""
        return torch.tensor(self.alpha, dtype=torch.float32, device=device)


# ==============================================================================
# NSGA-II CORE FUNCTIONS
# ==============================================================================

def dominates(p: np.ndarray, q: np.ndarray) -> bool:
    """
    Check if solution p dominates solution q (minimization).

    Args:
        p: Fitness array [f1, f2, f3] - lower is better
        q: Fitness array [f1, f2, f3] - lower is better

    Returns:
        True if p dominates q
    """
    no_worse = np.all(p <= q)
    strictly_better = np.any(p < q)
    return no_worse and strictly_better


def non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """
    Perform non-dominated sorting to assign Pareto ranks.

    Args:
        population: List of individuals

    Returns:
        List of fronts, each front is list of indices
    """
    n = len(population)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]

    fronts = [[]]

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if population[i].fitness is None or population[j].fitness is None:
                continue

            if dominates(population[i].fitness, population[j].fitness):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(population[j].fitness, population[i].fitness):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    # Find first front
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            if i not in fronts[0]:
                fronts[0].append(i)

    # Find subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    # Remove empty front
    if fronts and not fronts[-1]:
        fronts = fronts[:-1]

    return fronts


def calculate_crowding_distance(population: List[Individual], front_indices: List[int]) -> Dict[int, float]:
    """
    Calculate crowding distance for solutions in a front.

    Args:
        population: Full population
        front_indices: Indices of solutions in the front

    Returns:
        Dictionary mapping index to crowding distance
    """
    if len(front_indices) <= 2:
        return {i: float('inf') for i in front_indices}

    n_objectives = len(population[front_indices[0]].fitness)
    distances = {i: 0.0 for i in front_indices}

    for obj_idx in range(n_objectives):
        sorted_indices = sorted(
            front_indices,
            key=lambda i: population[i].fitness[obj_idx]
        )

        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Get objective range
        obj_values = [population[i].fitness[obj_idx] for i in sorted_indices]
        obj_range = obj_values[-1] - obj_values[0]

        if obj_range == 0:
            continue

        # Calculate distance for intermediate solutions
        for i in range(1, len(sorted_indices) - 1):
            distances[sorted_indices[i]] += (
                (obj_values[i + 1] - obj_values[i - 1]) / obj_range
            )

    return distances


def assign_crowding_distance(population: List[Individual], fronts: List[List[int]]) -> None:
    """
    Assign crowding distance to all individuals.

    Args:
        population: List of individuals
        fronts: List of fronts
    """
    for front in fronts:
        distances = calculate_crowding_distance(population, front)
        for idx, dist in distances.items():
            population[idx].crowding_distance = dist


# ==============================================================================
# GENETIC OPERATORS
# ==============================================================================

def tournament_selection(population: List[Individual], tournament_size: int = 2) -> Individual:
    """
    Tournament selection based on rank and crowding distance.

    Args:
        population: Population to select from
        tournament_size: Number in tournament

    Returns:
        Selected individual
    """
    candidates = random.sample(range(len(population)), tournament_size)
    best = candidates[0]

    for candidate in candidates[1:]:
        if population[candidate].rank < population[best].rank:
            best = candidate
        elif population[candidate].rank == population[best].rank:
            if population[candidate].crowding_distance > population[best].crowding_distance:
                best = candidate

    return copy.deepcopy(population[best])


def crossover(parent1: Individual, parent2: Individual, config: NSGA2Config) -> Tuple[Individual, Individual]:
    """
    Crossover for mask genes and alpha.

    Args:
        parent1: Parent 1
        parent2: Parent 2
        config: Configuration

    Returns:
        Two offspring
    """
    # Uniform crossover for mask genes
    mask1 = parent1.mask_genes.copy()
    mask2 = parent2.mask_genes.copy()

    if random.random() < config.crossover_rate:
        # Swap random rows
        n_rows = mask1.shape[0]
        swap_rows = random.sample(range(n_rows), random.randint(1, n_rows // 2))
        for row in swap_rows:
            mask1[row], mask2[row] = mask2[row].copy(), mask1[row].copy()

    # Arithmetic crossover for alpha
    alpha1 = parent1.alpha
    alpha2 = parent2.alpha

    if random.random() < config.crossover_rate:
        alpha = random.random()
        alpha1 = alpha * parent1.alpha + (1 - alpha) * parent2.alpha
        alpha2 = (1 - alpha) * parent1.alpha + alpha * parent2.alpha

    # Clamp alpha
    alpha1 = np.clip(alpha1, config.min_alpha, config.max_alpha)
    alpha2 = np.clip(alpha2, config.min_alpha, config.max_alpha)

    child1 = Individual(mask_genes=mask1, alpha=alpha1)
    child2 = Individual(mask_genes=mask2, alpha=alpha2)

    return child1, child2


def mutate(individual: Individual, config: NSGA2Config) -> Individual:
    """
    Mutation for mask genes and alpha.

    Args:
        individual: Individual to mutate
        config: Configuration

    Returns:
        Mutated individual
    """
    mutated = copy.deepcopy(individual)

    # Mutation for mask genes (bit flip)
    if random.random() < config.mutation_rate:
        n_genes = mutated.mask_genes.size
        n_flips = max(1, int(n_genes * 0.05))  # 5% mutation rate

        flat_mask = mutated.mask_genes.flatten()
        flip_indices = random.sample(range(n_genes), min(n_flips, n_genes))
        for idx in flip_indices:
            flat_mask[idx] = 1 - flat_mask[idx]
        mutated.mask_genes = flat_mask.reshape(mutated.mask_genes.shape)

        # Enforce density constraints
        current_density = mutated.mask_genes.mean()
        if current_density < config.min_density:
            # Add more 1s
            flat_mask = mutated.mask_genes.flatten()
            zero_indices = np.where(flat_mask == 0)[0]
            n_add = int(n_genes * (config.min_density - current_density))
            if len(zero_indices) > 0 and n_add > 0:
                add_indices = random.sample(list(zero_indices), min(n_add, len(zero_indices)))
                for idx in add_indices:
                    flat_mask[idx] = 1
                mutated.mask_genes = flat_mask.reshape(mutated.mask_genes.shape)
        elif current_density > config.max_density:
            # Remove some 1s
            flat_mask = mutated.mask_genes.flatten()
            one_indices = np.where(flat_mask == 1)[0]
            n_remove = int(n_genes * (current_density - config.max_density))
            if len(one_indices) > 0 and n_remove > 0:
                remove_indices = random.sample(list(one_indices), min(n_remove, len(one_indices)))
                for idx in remove_indices:
                    flat_mask[idx] = 0
                mutated.mask_genes = flat_mask.reshape(mutated.mask_genes.shape)

    # Mutation for alpha (Gaussian)
    if random.random() < config.mutation_rate:
        alpha_noise = np.random.randn() * 0.1
        mutated.alpha = np.clip(mutated.alpha + alpha_noise, config.min_alpha, config.max_alpha)

    return mutated


# ==============================================================================
# POPULATION INITIALIZATION
# ==============================================================================

def initialize_population(config: NSGA2Config) -> List[Individual]:
    """
    Initialize random population.

    Args:
        config: Configuration

    Returns:
        List of individuals
    """
    population = []
    block_h, block_w = config.block_grid_h, config.block_grid_w

    for _ in range(config.population_size):
        # Random binary mask genes
        mask_genes = np.random.binomial(1, config.initial_density, (block_h, block_w)).astype(np.uint8)

        # Random alpha
        alpha = np.random.uniform(config.initial_alpha * 0.5, config.initial_alpha * 1.5)
        alpha = np.clip(alpha, config.min_alpha, config.max_alpha)

        individual = Individual(mask_genes=mask_genes, alpha=alpha)
        population.append(individual)

    return population


# ==============================================================================
# FITNESS EVALUATION
# ==============================================================================

def evaluate_individual(
    individual: Individual,
    cover_images: torch.Tensor,
    messages: torch.Tensor,
    encoder_decoder: nn.Module,
    discriminator: nn.Module,
    config: NSGA2Config
) -> np.ndarray:
    """
    Evaluate fitness for one individual.

    Args:
        individual: Individual to evaluate
        cover_images: Cover images [B, 3, H, W]
        messages: Messages [B, message_length]
        encoder_decoder: Frozen encoder-decoder model
        discriminator: Frozen discriminator
        config: Configuration

    Returns:
        Fitness array [f1, f2, f3] - lower is better for all
    """
    encoder_decoder.eval()
    discriminator.eval()

    device = config.device
    B = cover_images.shape[0]
    H, W = config.image_height, config.image_width
    message_length = config.message_length

    with torch.no_grad():
        # Create mask and alpha tensors
        mask = individual.to_mask_tensor(H, W, device)
        alpha = individual.to_alpha_tensor(device)

        # Run encoder-decoder
        encoded_images, noised_images, decoded_messages = encoder_decoder(
            cover_images, messages, mask=mask, alpha=alpha
        )

        # Calculate metrics
        # PSNR - higher is better, normalize to [0, 1]
        psnr = calculate_psnr(cover_images, encoded_images, data_range=2.0)
        psnr_norm = torch.clamp(psnr / 50.0, 0, 1).mean()  # Assume 50dB is excellent

        # SSIM - already in [0, 1]
        ssim = calculate_ssim(cover_images, encoded_images, data_range=2.0).mean()

        # Imperceptibility (higher is better)
        imperceptibility = (config.psnr_weight * psnr_norm + config.ssim_weight * ssim) / (
            config.psnr_weight + config.ssim_weight)

        # BPP (bits per pixel) - higher is better
        bpp = calculate_bpp(message_length, H, W)
        bpp_normalized = bpp * 100  # Scale for better numerical handling

        # BER - check constraint
        ber = calculate_ber(decoded_messages, messages).mean()

        # Security - lower detection probability is better
        security = calculate_security_score(discriminator, encoded_images, cover_images).mean()

        # Objective 1: Imperceptibility (minimize)
        f1 = -imperceptibility

        # Objective 2: Payload (minimize - so we minimize negative of bpp)
        f2 = -bpp_normalized

        # Objective 3: Security (minimize detection probability)
        f3 = security

        # Apply penalty if BER > threshold
        if ber > config.ber_threshold:
            penalty = 10.0 * (ber - config.ber_threshold)
            f1 += penalty
            f2 += penalty
            f3 += penalty

        fitness = np.array([f1, f2, f3])

    return fitness


def evaluate_population(
    population: List[Individual],
    cover_images: torch.Tensor,
    messages: torch.Tensor,
    encoder_decoder: nn.Module,
    discriminator: nn.Module,
    config: NSGA2Config
) -> None:
    """
    Evaluate fitness for all individuals in population.

    Args:
        population: Population to evaluate
        cover_images: Cover images [B, 3, H, W]
        messages: Messages [B, message_length]
        encoder_decoder: Frozen encoder-decoder model
        discriminator: Frozen discriminator
        config: Configuration
    """
    for individual in population:
        if individual.fitness is None:
            individual.fitness = evaluate_individual(
                individual, cover_images, messages,
                encoder_decoder, discriminator, config
            )


# ==============================================================================
# NSGA-II MAIN LOOP
# ==============================================================================

def create_offspring_population(population: List[Individual], config: NSGA2Config) -> List[Individual]:
    """
    Create offspring through selection, crossover, and mutation.

    Args:
        population: Current population
        config: Configuration

    Returns:
        Offspring population
    """
    offspring = []

    while len(offspring) < config.population_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        child1, child2 = crossover(parent1, parent2, config)

        child1 = mutate(child1, config)
        child2 = mutate(child2, config)

        offspring.append(child1)
        if len(offspring) < config.population_size:
            offspring.append(child2)

    return offspring[:config.population_size]


def select_next_generation(
    population: List[Individual],
    offspring: List[Individual],
    config: NSGA2Config
) -> List[Individual]:
    """
    Select next generation using NSGA-II selection.

    Args:
        population: Current population
        offspring: Offspring population
        config: Configuration

    Returns:
        Selected next generation
    """
    combined = population + offspring

    # Non-dominated sorting
    fronts = non_dominated_sort(combined)

    # Assign crowding distance
    assign_crowding_distance(combined, fronts)

    # Select best individuals
    new_population = []
    front_idx = 0

    while len(new_population) + len(fronts[front_idx]) <= config.population_size:
        new_population.extend([combined[i] for i in fronts[front_idx]])
        front_idx += 1
        if front_idx >= len(fronts):
            break

    # Handle last front
    remaining = config.population_size - len(new_population)
    if remaining > 0 and front_idx < len(fronts):
        last_front = sorted(
            fronts[front_idx],
            key=lambda i: combined[i].crowding_distance,
            reverse=True
        )
        new_population.extend([combined[i] for i in last_front[:remaining]])

    return new_population[:config.population_size]


def run_nsga2(
    cover_images: torch.Tensor,
    messages: torch.Tensor,
    encoder_decoder: nn.Module,
    discriminator: nn.Module,
    config: NSGA2Config = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[Individual], List[int]]:
    """
    Run NSGA-II optimization.

    Args:
        cover_images: Cover images [B, 3, H, W]
        messages: Messages [B, message_length]
        encoder_decoder: Frozen encoder-decoder model
        discriminator: Frozen discriminator
        config: NSGA-II configuration
        verbose: Print progress
        progress_callback: Optional callback function(generation, population, pareto_front)

    Returns:
        Tuple of (final_population, pareto_front_indices)
    """
    if config is None:
        config = NSGA2Config()

    if verbose:
        logging.info(f"Starting NSGA-II: pop_size={config.population_size}, generations={config.n_generations}")
        logging.info(f"Block mask size: {config.block_grid_h}x{config.block_grid_w}")
        logging.info(f"Objectives: Imperceptibility, Payload, Security")

    # Initialize population
    population = initialize_population(config)

    # Main evolutionary loop
    for generation in range(config.n_generations):
        # Evaluate fitness
        evaluate_population(population, cover_images, messages, encoder_decoder, discriminator, config)

        # Create offspring
        offspring = create_offspring_population(population, config)

        # Evaluate offspring
        evaluate_population(offspring, cover_images, messages, encoder_decoder, discriminator, config)

        # Select next generation
        population = select_next_generation(population, offspring, config)

        # Progress
        if verbose and (generation + 1) % 10 == 0:
            fronts = non_dominated_sort(population)
            pareto_front = [population[i] for i in fronts[0]]

            avg_f1 = np.mean([ind.fitness[0] for ind in pareto_front])
            avg_f2 = np.mean([ind.fitness[1] for ind in pareto_front])
            avg_f3 = np.mean([ind.fitness[2] for ind in pareto_front])

            logging.info(f"Gen {generation+1}/{config.n_generations}: "
                        f"f1(imp)={avg_f1:.4f}, f2(payload)={avg_f2:.4f}, f3(security)={avg_f3:.4f}")

        if progress_callback:
            fronts = non_dominated_sort(population)
            progress_callback(generation, population, fronts[0])

    # Final evaluation
    evaluate_population(population, cover_images, messages, encoder_decoder, discriminator, config)
    fronts = non_dominated_sort(population)
    pareto_front_indices = fronts[0]

    if verbose:
        logging.info(f"Optimization complete. Pareto front size: {len(pareto_front_indices)}")

    return population, pareto_front_indices


# ==============================================================================
# PARETO FRONT ANALYSIS
# ==============================================================================

def analyze_pareto_front(
    population: List[Individual],
    pareto_indices: List[int],
    config: NSGA2Config
) -> Dict:
    """
    Analyze Pareto front solutions.

    Args:
        population: Full population
        pareto_indices: Pareto front indices
        config: Configuration

    Returns:
        Analysis dictionary
    """
    pareto_solutions = [population[i] for i in pareto_indices]

    f1 = [sol.fitness[0] for sol in pareto_solutions]
    f2 = [sol.fitness[1] for sol in pareto_solutions]
    f3 = [sol.fitness[2] for sol in pareto_solutions]

    alphas = [sol.alpha for sol in pareto_solutions]
    densities = [sol.mask_genes.mean() for sol in pareto_solutions]

    return {
        'n_solutions': len(pareto_solutions),
        'imperceptibility': {'min': np.min(-np.array(f1)), 'max': np.max(-np.array(f1)), 'mean': np.mean(-np.array(f1))},
        'payload_bpp': {'min': np.min(-np.array(f2)), 'max': np.max(-np.array(f2)), 'mean': np.mean(-np.array(f2))},
        'security': {'min': np.min(f3), 'max': np.max(f3), 'mean': np.mean(f3)},
        'alpha': {'min': np.min(alphas), 'max': np.max(alphas), 'mean': np.mean(alphas)},
        'mask_density': {'min': np.min(densities), 'max': np.max(densities), 'mean': np.mean(densities)}
    }


def select_best_solution(
    population: List[Individual],
    pareto_indices: List[int],
    priority: str = 'balanced'
) -> Individual:
    """
    Select best solution from Pareto front based on priority.

    Args:
        population: Full population
        pareto_indices: Pareto front indices
        priority: 'imperceptibility', 'payload', 'security', 'balanced'

    Returns:
        Best individual
    """
    pareto_solutions = [population[i] for i in pareto_indices]

    if priority == 'imperceptibility':
        # Best visual quality (highest -f1)
        best_idx = np.argmax([-sol.fitness[0] for sol in pareto_solutions])
    elif priority == 'payload':
        # Highest payload (highest -f2)
        best_idx = np.argmax([-sol.fitness[1] for sol in pareto_solutions])
    elif priority == 'security':
        # Best security (lowest f3)
        best_idx = np.argmin([sol.fitness[2] for sol in pareto_solutions])
    else:  # balanced
        # Closest to ideal point
        ideal = np.array([-1, -1, 0])  # Perfect: high imperceptibility, high payload, low detection
        distances = []
        for sol in pareto_solutions:
            dist = np.linalg.norm(ideal - sol.fitness)
            distances.append(dist)
        best_idx = np.argmin(distances)

    return pareto_solutions[best_idx]


def print_pareto_solutions(
    population: List[Individual],
    pareto_indices: List[int],
    config: NSGA2Config
) -> None:
    """
    Print Pareto front solutions with all metrics.

    Args:
        population: Full population
        pareto_indices: Pareto front indices
        config: Configuration
    """
    print("\n" + "=" * 100)
    print("PARETO FRONT SOLUTIONS")
    print("=" * 100)
    print(f"{'IDX':<5} {'PSNR':<8} {'SSIM':<8} {'BPP':<8} {'Security':<10} {'Alpha':<8} {'Density':<10}")
    print("-" * 100)

    for i, idx in enumerate(pareto_indices):
        sol = population[idx]
        # Convert fitness back to original values
        psnr = 50.0 * max(0, -sol.fitness[0])  # Approximate
        ssim = max(0, -sol.fitness[0])  # Simplified
        bpp = -sol.fitness[1] / 100.0  # Approximate
        security = sol.fitness[2]
        density = sol.mask_genes.mean()

        print(f"{i:<5} {psnr:<8.2f} {ssim:<8.4f} {bpp:<8.4f} {security:<10.4f} {sol.alpha:<8.4f} {density:<10.4f}")

    print("=" * 100 + "\n")