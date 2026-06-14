"""
Integration module for NSGA-II with HiDDeN steganography system.

This module provides functions to:
1. Load trained HiDDeN models for stego image generation
2. Integrate with real steganalysis tools
3. Save and visualize Pareto front results
4. Customize optimization for specific datasets

Author: NSGA-II Integration for Steganography
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable
import json
import pickle
import logging
from dataclasses import dataclass, asdict

from nsga2_optimizer import (
    Individual, NSGA2Config,
    run_nsga2, analyze_pareto_front, select_best_solution, print_pareto_solutions,
)

from metrics import calculate_psnr, calculate_ssim, calculate_payload

# ==============================================================================
# MODEL LOADER AND STEGANALYZER
# ==============================================================================

def load_trained_hidden_model(model_path: str, config_path: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load a trained HiDDeN model for stego image generation.

    Args:
        model_path: Path to the saved model checkpoint (.pyt file)
        config_path: Path to the saved configuration pickle file
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        Loaded encoder-decoder model

    TODO: Update this to match your actual model loading pattern
    """
    import utils
    from model.hidden import Hidden
    from noise_layers.noiser import Noiser
    from options import HiDDenConfiguration

    # Load configuration
    train_options, hidden_config, noise_config = utils.load_options(config_path)

    # Create model
    noiser = Noiser(noise_config, device)
    device_obj = torch.device(device)

    # Note: This creates a new model; actual loading requires matching the architecture
    # For now, return None and use the placeholder
    logging.warning("Model loading requires matching architecture - using placeholder")

    return None


class SteganalyzerWrapper:
    """
    Wrapper for steganalysis models.

    This provides a unified interface for different types of steganalyzers:
    - CNN-based (SRNet, XuNet, etc.)
    - Feature-based (SRB, SPAM, etc.)
    - Ensemble classifiers

    TODO: Replace placeholder methods with actual steganalysis implementation
    """

    def __init__(self, model: Optional[torch.nn.Module] = None, model_type: str = 'placeholder'):
        """
        Initialize steganalyzer wrapper.

        Args:
            model: Trained steganalysis model (if available)
            model_type: Type of steganalyzer ('cnn', 'feature', 'ensemble', 'placeholder')
        """
        self.model = model
        self.model_type = model_type
        self.device = next(model.parameters()).device if model is not None else torch.device('cpu')

    def detect(self, image: np.ndarray) -> float:
        """
        Detect if an image contains hidden data.

        Args:
            image: Image to analyze (H x W x C) in range [0, 255]

        Returns:
            Detection probability (0-1). Lower = more secure.
        """
        if self.model is None or self.model_type == 'placeholder':
            # Use placeholder
            return self._placeholder_detect(image)

        # TODO: Implement real detection
        return self._placeholder_detect(image)

    def detect_batch(self, images: List[np.ndarray]) -> List[float]:
        """
        Detect hidden data in a batch of images.

        Args:
            images: List of images to analyze

        Returns:
            List of detection probabilities
        """
        return [self.detect(img) for img in images]

    def _placeholder_detect(self, image: np.ndarray) -> float:
        """
        Placeholder detection based on statistical features.

        This is a simplified version - replace with actual steganalysis.
        """
        # Calculate statistical features
        img_float = image.astype(np.float64)

        # Variance of differences (simplified residual)
        if len(img_float.shape) == 3:
            # Convert to grayscale for analysis
            gray = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
        else:
            gray = img_float

        # Horizontal and vertical gradients
        h_grad = np.abs(np.diff(gray, axis=1))
        v_grad = np.abs(np.diff(gray, axis=0))

        # Statistical features that might indicate steganography
        mean_h = np.mean(h_grad)
        std_h = np.std(h_grad)
        mean_v = np.mean(v_grad)
        std_v = np.std(v_grad)

        # Combined feature (simplified)
        detection = min(1.0, (std_h + std_v) / 50)

        return detection


# ==============================================================================
# COVER IMAGE LOADER
# ==============================================================================

class CoverImageLoader:
    """
    Load and manage cover images from dataset for NSGA-II optimization.

    This handles different dataset structures:
    - ImageFolder format (train/val subdirectories)
    - Custom directory structure
    - Pre-loaded numpy arrays
    """

    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (128, 128)):
        """
        Initialize cover image loader.

        Args:
            data_dir: Directory containing images
            image_size: Target size (H, W) for images
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.image_paths = []

    def load_from_folder(self, split: str = 'val') -> List[np.ndarray]:
        """
        Load images from ImageFolder-style directory.

        Expected structure:
            data_dir/
                train/
                    class1/
                        image1.jpg
                    class2/
                val/
                    ...

        Args:
            split: 'train' or 'val'

        Returns:
            List of images as numpy arrays (H x W x C)
        """
        from torchvision import transforms
        from PIL import Image
        import os

        split_dir = self.data_dir / split
        images = []

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

        # Walk through all image files
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.image_size)
                        img_array = np.array(img)
                        images.append(img_array)
                    except Exception as e:
                        logging.warning(f"Failed to load {img_path}: {e}")

        logging.info(f"Loaded {len(images)} images from {split_dir}")
        return images

    def load_sample(self, n_samples: int = 10, split: str = 'val') -> List[np.ndarray]:
        """
        Load a random sample of images.

        Args:
            n_samples: Number of images to load
            split: 'train' or 'val'

        Returns:
            List of images
        """
        all_images = self.load_from_folder(split)

        if len(all_images) <= n_samples:
            return all_images

        # Random sample
        indices = np.random.choice(len(all_images), n_samples, replace=False)
        return [all_images[i] for i in indices]


# ==============================================================================
# CUSTOM OBJECTIVE FUNCTIONS
# ==============================================================================

class CustomObjectiveEvaluator:
    """
    Custom objective evaluator that allows defining custom objective functions.

    This is useful when:
    - You want to use different imperceptibility metrics
    - You have a custom payload calculation
    - You have a real steganalysis model
    """

    def __init__(self,
                 imperceptibility_func: Optional[Callable] = None,
                 payload_func: Optional[Callable] = None,
                 security_func: Optional[Callable] = None):
        """
        Initialize custom evaluator.

        Args:
            imperceptibility_func: Custom function(cover, stego) -> score
            payload_func: Custom function(mask, message_length) -> bpp
            security_func: Custom function(cover, stego) -> detection_prob
        """
        self.imperceptibility_func = imperceptibility_func
        self.payload_func = payload_func
        self.security_func = security_func

    def evaluate(self, individual: Individual, cover_image: np.ndarray,
                 stego_image: np.ndarray, config: NSGA2Config) -> np.ndarray:
        """
        Evaluate objectives with custom functions.

        Args:
            individual: Individual to evaluate
            cover_image: Cover image
            stego_image: Generated stego image
            config: Configuration

        Returns:
            Fitness array [imperceptibility, payload, security]
        """
        # Imperceptibility
        if self.imperceptibility_func is not None:
            imperceptibility = self.imperceptibility_func(cover_image, stego_image)
        else:
            imperceptibility = self._default_imperceptibility(cover_image, stego_image, config)

        # Payload
        if self.payload_func is not None:
            payload = self.payload_func(individual.embedding_mask, config.message_length)
        else:
            payload = calculate_payload(individual.embedding_mask, config.message_length)
        payload_normalized = payload / config.max_mask_ratio

        # Security
        if self.security_func is not None:
            security = self.security_func(cover_image, stego_image)
        else:
            from nsga2_optimizer import placeholder_steganalyzer
            security = placeholder_steganalyzer(cover_image, stego_image)

        return np.array([imperceptibility, payload_normalized, 1.0 - security])

    def _default_imperceptibility(self, cover: np.ndarray, stego: np.ndarray,
                                  config: NSGA2Config) -> float:
        """Default imperceptibility calculation"""
        psnr = calculate_psnr(cover, stego)
        ssim = calculate_ssim(cover, stego)

        if psnr == float('inf'):
            psnr = 50.0

        psnr_norm = min(1.0, psnr / 50.0)
        ssim_norm = ssim

        return (config.psnr_weight * psnr_norm + config.ssim_weight * ssim_norm) / \
               (config.psnr_weight + config.ssim_weight)


# ==============================================================================
# RESULTS SAVER AND VISUALIZATION
# ==============================================================================

@dataclass
class OptimizationResult:
    """Container for NSGA-II optimization results"""
    config: Dict
    pareto_analysis: Dict
    best_by_priority: Dict
    population_size: int
    n_generations: int
    final_pareto_size: int


class ResultsManager:
    """
    Save and visualize NSGA-II optimization results.
    """

    def __init__(self, output_dir: str):
        """
        Initialize results manager.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, population: List[Individual], pareto_indices: List[int],
                     config: NSGA2Config, analysis: Dict, best_solutions: Dict,
                     output_prefix: str = 'nsga2_results') -> None:
        """
        Save optimization results to files.

        Args:
            population: Final population
            pareto_indices: Pareto front indices
            config: NSGA-II configuration
            analysis: Pareto front analysis
            best_solutions: Best solutions by priority
            output_prefix: Prefix for output files
        """
        # Save results as JSON
        results = {
            'config': asdict(config),
            'analysis': analysis,
            'best_solutions': {
                priority: {
                    'fitness': sol.fitness.tolist(),
                    'mask_ratio': float(np.sum(sol.embedding_mask) / sol.embedding_mask.size),
                    'params': sol.embedding_params.tolist()
                }
                for priority, sol in best_solutions.items()
            },
            'n_pareto_solutions': len(pareto_indices)
        }

        json_path = self.output_dir / f'{output_prefix}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {json_path}")

        # Save population data as pickle
        pop_data = []
        for i, ind in enumerate(population):
            pop_data.append({
                'index': i,
                'fitness': ind.fitness,
                'rank': ind.rank,
                'crowding_distance': ind.crowding_distance,
                'mask': ind.embedding_mask,
                'params': ind.embedding_params,
                'is_pareto': i in pareto_indices
            })

        pickle_path = self.output_dir / f'{output_prefix}_population.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(pop_data, f)

        logging.info(f"Population data saved to {pickle_path}")

        # Save Pareto front masks as numpy files
        for priority, sol in best_solutions.items():
            mask_path = self.output_dir / f'{output_prefix}_mask_{priority}.npy'
            np.save(mask_path, sol.embedding_mask)

        logging.info(f"Mask files saved to {self.output_dir}")

    def visualize_pareto_front_2d(self, population: List[Individual],
                                   pareto_indices: List[int],
                                   obj1_name: str, obj2_name: str,
                                   output_name: str = 'pareto_2d.png') -> None:
        """
        Create 2D visualization of Pareto front (two objectives).

        Args:
            population: Population
            pareto_indices: Pareto front indices
            obj1_name: Name of first objective
            obj2_name: Name of second objective
            output_name: Output filename
        """
        try:
            import matplotlib.pyplot as plt

            pareto_sols = [population[i] for i in pareto_indices]

            obj1_map = {'imperceptibility': 0, 'payload': 1, 'security': 2}
            obj2_map = {'imperceptibility': 0, 'payload': 1, 'security': 2}

            obj1_idx = obj1_map.get(obj1_name.lower(), 0)
            obj2_idx = obj2_map.get(obj2_name.lower(), 1)

            x = [sol.fitness[obj1_idx] for sol in pareto_sols]
            y = [sol.fitness[obj2_idx] for sol in pareto_sols]

            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, c='blue', s=100, alpha=0.7, edgecolors='black')
            plt.xlabel(obj1_name)
            plt.ylabel(obj2_name)
            plt.title(f'Pareto Front: {obj1_name} vs {obj2_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_path = self.output_dir / output_name
            plt.savefig(output_path, dpi=150)
            plt.close()

            logging.info(f"2D Pareto plot saved to {output_path}")

        except ImportError:
            logging.warning("matplotlib not available - skipping visualization")


# ==============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ==============================================================================

def run_optimization_with_custom_config(
    data_dir: str,
    model_path: Optional[str] = None,
    steganalyzer_path: Optional[str] = None,
    config: Optional[NSGA2Config] = None,
    output_dir: str = './nsga2_results'
) -> Tuple[List[Individual], List[int]]:
    """
    Complete NSGA-II optimization workflow.

    This function demonstrates the complete workflow:
    1. Load cover images from dataset
    2. Load model (optional)
    3. Load steganalyzer (optional)
    4. Run NSGA-II
    5. Analyze results
    6. Save outputs

    Args:
        data_dir: Directory containing cover images
        model_path: Path to trained HiDDeN model (optional)
        steganalyzer_path: Path to steganalysis model (optional)
        config: NSGA-II configuration
        output_dir: Directory to save results

    Returns:
        Tuple of (final_population, pareto_indices)
    """
    # Default configuration
    if config is None:
        config = NSGA2Config(
            population_size=50,
            n_generations=100,
            image_height=128,
            image_width=128,
            message_length=30
        )

    # Load cover images
    loader = CoverImageLoader(data_dir, image_size=(config.image_height, config.image_width))
    cover_images = loader.load_sample(n_samples=10, split='val')

    if not cover_images:
        raise ValueError(f"No images found in {data_dir}")

    # Load model
    model = None
    if model_path:
        try:
            model = load_trained_hidden_model(model_path, model_path.replace('.pyt', '.pickle'))
        except Exception as e:
            logging.warning(f"Failed to load model: {e}")

    # Load steganalyzer
    steganalyzer = None
    if steganalyzer_path:
        try:
            steganalyzer = torch.load(steganalyzer_path)
        except Exception as e:
            logging.warning(f"Failed to load steganalyzer: {e}")

    # Run NSGA-II
    logging.info("Starting NSGA-II optimization...")
    population, pareto_indices = run_nsga2(
        cover_images=cover_images,
        model=model,
        config=config,
        verbose=True
    )

    # Analyze results
    analysis = analyze_pareto_front(population, pareto_indices)

    # Get best solutions by priority
    best_solutions = {}
    for priority in ['imperceptibility', 'payload', 'security', 'balanced']:
        best_solutions[priority] = select_best_solution(
            population, pareto_indices, priority=priority
        )

    # Save results
    results_manager = ResultsManager(output_dir)
    results_manager.save_results(
        population, pareto_indices, config, analysis, best_solutions
    )

    # Create visualizations
    results_manager.visualize_pareto_front_2d(
        population, pareto_indices, 'imperceptibility', 'payload'
    )
    results_manager.visualize_pareto_front_2d(
        population, pareto_indices, 'imperceptibility', 'security'
    )
    results_manager.visualize_pareto_front_2d(
        population, pareto_indices, 'payload', 'security'
    )

    return population, pareto_indices


# ==============================================================================
# PARAMETER ADJUSTMENT GUIDE
# ==============================================================================

PARAMETER_GUIDE = """
NSGA-II PARAMETER ADJUSTMENT GUIDE
==================================

1. POPULATION SIZE
   - Larger: Better exploration, more computational cost
   - Smaller: Faster, may miss good solutions
   - Recommended: 30-100 for most problems
   - Adjust based on: Available GPU memory, time constraints

2. NUMBER OF GENERATIONS
   - More: Better convergence, more computational cost
   - Less: Faster, may not converge
   - Recommended: 50-200 for most problems
   - Adjust based on: Complexity of objectives, time constraints

3. CROSSOVER RATE
   - Higher: More exploration, faster convergence
   - Lower: More exploitation, slower convergence
   - Recommended: 0.8-0.95

4. MUTATION RATE
   - Higher: More diversity, slower convergence
   - Lower: Less diversity, faster convergence
   - Recommended: 0.05-0.2

5. IMAGE PARAMETERS
   - image_height/image_width: Match your dataset
   - message_length: Longer = more data but more distortion

6. MASK PARAMETERS
   - initial_mask_ratio: Starting fraction of embedding pixels
   - min_mask_ratio/max_mask_ratio: Bounds on embedding capacity

7. OBJECTIVE WEIGHTS
   - psnr_weight: Higher if PSNR is important
   - ssim_weight: Higher if SSIM is important
   - Adjust based on: Which objective matters most for your application

DATASET-SPECIFIC ADJUSTMENTS
============================

For different datasets, consider:

1. Image Size: Match your dataset dimensions
2. Message Length: Based on payload requirements
3. Noise Layers: Adjust if training with different noise
4. Batch Size: Based on GPU memory

EMBEDDING METHOD INTEGRATION
============================

To integrate with your embedding method:

1. Replace generate_stego_image_with_model() with your method
2. Replace placeholder_steganalyzer with real steganalysis
3. Adjust objective weights based on priorities
"""

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Example main function showing complete workflow.
    """
    import argparse

    parser = argparse.ArgumentParser(description='NSGA-II for Steganography Optimization')
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='Directory containing cover images')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to trained HiDDeN model')
    parser.add_argument('--steganalyzer', '-s', type=str, default=None,
                        help='Path to steganalysis model')
    parser.add_argument('--output', '-o', type=str, default='./nsga2_results',
                        help='Output directory for results')
    parser.add_argument('--pop-size', '-p', type=int, default=50,
                        help='Population size')
    parser.add_argument('--generations', '-g', type=int, default=100,
                        help='Number of generations')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size (square)')
    parser.add_argument('--message-length', type=int, default=30,
                        help='Message length in bits')

    args = parser.parse_args()

    # Configure NSGA-II
    config = NSGA2Config(
        population_size=args.pop_size,
        n_generations=args.generations,
        image_height=args.image_size,
        image_width=args.image_size,
        message_length=args.message_length
    )

    # Run optimization
    population, pareto_indices = run_optimization_with_custom_config(
        data_dir=args.data_dir,
        model_path=args.model,
        steganalyzer_path=args.steganalyzer,
        config=config,
        output_dir=args.output
    )

    print(PARAMETER_GUIDE)


if __name__ == '__main__':
    main()