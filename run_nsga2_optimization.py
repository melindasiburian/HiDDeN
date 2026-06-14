"""
Run NSGA-II Optimization for Mask-Aware Steganography

This script:
1. Loads a trained encoder-decoder and discriminator
2. Freezes model parameters
3. Loads a batch of cover images and messages
4. Runs NSGA-II to optimize mask and alpha
5. Saves and displays Pareto front solutions

Usage:
    python run_nsga2_optimization.py --checkpoint-folder /path/to/checkpoint --data-dir /path/to/images

The checkpoint folder should contain:
- A trained encoder-decoder model
- A trained discriminator model

The data directory should contain images for evaluation.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from model.encoder_decoder import EncoderDecoder
from model.discriminator import Discriminator
import utils
from nsga2_optimizer import (
    NSGA2Config, run_nsga2, analyze_pareto_front,
    select_best_solution, print_pareto_solutions
)
from metrics import calculate_psnr, calculate_ssim, calculate_ber


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_trained_models(checkpoint_folder: str, device: torch.device):
    """
    Load trained encoder-decoder and discriminator models.

    Args:
        checkpoint_folder: Path to checkpoint folder
        device: Device to load models on

    Returns:
        Tuple of (encoder_decoder, discriminator)
    """
    # Load options and config
    options_file = os.path.join(checkpoint_folder, 'options-and-config.pickle')
    if not os.path.exists(options_file):
        raise FileNotFoundError(f"Options file not found: {options_file}")

    train_options, hidden_config, noise_config = utils.load_options(options_file)

    # Create models
    noiser = Noiser(noise_config, device)
    encoder_decoder = EncoderDecoder(hidden_config, noiser).to(device)
    discriminator = Discriminator(hidden_config).to(device)

    # Find latest checkpoint
    checkpoint_files = sorted(Path(checkpoint_folder).glob("checkpoints/*.pyt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_folder}/checkpoints")

    latest_checkpoint = checkpoint_files[-1]
    logging.info(f"Loading checkpoint: {latest_checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    # Load encoder-decoder state
    encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])

    # Load discriminator state
    discriminator.load_state_dict(checkpoint['discrim-model'])

    # Freeze models
    for param in encoder_decoder.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    encoder_decoder.eval()
    discriminator.eval()

    return encoder_decoder, discriminator, hidden_config


def load_cover_images(
    data_dir: str,
    hidden_config: HiDDenConfiguration,
    batch_size: int = 4,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Load cover images for NSGA-II evaluation.

    Args:
        data_dir: Directory containing images (ImageFolder format)
        hidden_config: Model configuration
        batch_size: Number of images to load
        device: Device to load tensors on

    Returns:
        Cover images tensor [B, 3, H, W] in range [-1, 1]
    """
    from torchvision import datasets, transforms

    data_transform = transforms.Compose([
        transforms.CenterCrop((hidden_config.H, hidden_config.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Try to load from validation folder
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        val_dir = data_dir

    try:
        dataset = datasets.ImageFolder(val_dir, transform=data_transform)
        # Sample random images
        indices = np.random.choice(len(dataset), min(batch_size, len(dataset)), replace=False)
        images = [dataset[i][0] for i in indices]
        cover_images = torch.stack(images).to(device)
    except Exception as e:
        logging.warning(f"Failed to load images from {val_dir}: {e}")
        logging.info("Using dummy images for testing")
        cover_images = torch.randn(batch_size, 3, hidden_config.H, hidden_config.W, device=device)

    return cover_images


def generate_random_messages(
    batch_size: int,
    message_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generate random binary messages.

    Args:
        batch_size: Number of messages
        message_length: Length of each message
        device: Device

    Returns:
        Messages tensor [B, message_length]
    """
    messages = torch.randint(0, 2, (batch_size, message_length), dtype=torch.float32, device=device)
    return messages


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run NSGA-II optimization for steganography')

    parser.add_argument('--checkpoint-folder', '-c', type=str, required=True,
                        help='Path to trained model checkpoint folder')
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='Path to image data directory')
    parser.add_argument('--output-dir', '-o', type=str, default='./nsga2_results',
                        help='Output directory for results')

    # NSGA-II parameters
    parser.add_argument('--pop-size', '-p', type=int, default=30,
                        help='Population size')
    parser.add_argument('--generations', '-g', type=int, default=50,
                        help='Number of generations')
    parser.add_argument('--block-size', '-b', type=int, default=8,
                        help='Block size for mask')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--crossover-rate', type=float, default=0.9,
                        help='Crossover rate')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation rate')
    parser.add_argument('--ber-threshold', type=float, default=0.1,
                        help='BER threshold for constraint')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'nsga2.log')),
            logging.StreamHandler()
        ]
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load models
    logging.info(f"Loading models from {args.checkpoint_folder}")
    encoder_decoder, discriminator, hidden_config = load_trained_models(
        args.checkpoint_folder, device
    )
    logging.info(f"Models loaded successfully")
    logging.info(f"Image size: {hidden_config.H}x{hidden_config.W}")
    logging.info(f"Message length: {hidden_config.message_length}")

    # Load cover images
    logging.info(f"Loading images from {args.data_dir}")
    cover_images = load_cover_images(
        args.data_dir,
        hidden_config,
        batch_size=args.batch_size,
        device=device
    )
    logging.info(f"Loaded {cover_images.shape[0]} cover images")

    # Generate messages
    messages = generate_random_messages(
        cover_images.shape[0],
        hidden_config.message_length,
        device
    )

    # NSGA-II configuration
    config = NSGA2Config(
        population_size=args.pop_size,
        n_generations=args.generations,
        image_height=hidden_config.H,
        image_width=hidden_config.W,
        message_length=hidden_config.message_length,
        block_size=args.block_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        ber_threshold=args.ber_threshold,
        device=device
    )

    logging.info(f"NSGA-II Configuration:")
    logging.info(f"  Population size: {config.population_size}")
    logging.info(f"  Generations: {config.n_generations}")
    logging.info(f"  Block size: {config.block_size}")
    logging.info(f"  Block grid: {config.block_grid_h}x{config.block_grid_w}")

    # Run NSGA-II
    logging.info("Starting NSGA-II optimization...")
    population, pareto_indices = run_nsga2(
        cover_images=cover_images,
        messages=messages,
        encoder_decoder=encoder_decoder,
        discriminator=discriminator,
        config=config,
        verbose=True
    )

    # Analyze results
    analysis = analyze_pareto_front(population, pareto_indices, config)
    logging.info("\nPareto Front Analysis:")
    logging.info(f"  Number of solutions: {analysis['n_solutions']}")
    logging.info(f"  Imperceptibility: {analysis['imperceptibility']}")
    logging.info(f"  Payload (BPP): {analysis['payload_bpp']}")
    logging.info(f"  Security: {analysis['security']}")
    logging.info(f"  Alpha: {analysis['alpha']}")
    logging.info(f"  Mask density: {analysis['mask_density']}")

    # Print all Pareto solutions
    print_pareto_solutions(population, pareto_indices, config)

    # Select best solutions by priority
    print("\nBest solutions by priority:")
    for priority in ['imperceptibility', 'payload', 'security', 'balanced']:
        best = select_best_solution(population, pareto_indices, priority=priority)
        logging.info(f"  {priority}: alpha={best.alpha:.4f}, density={best.mask_genes.mean():.4f}")

    # Save results
    results_file = os.path.join(args.output_dir, 'nsga2_results.txt')
    with open(results_file, 'w') as f:
        f.write("NSGA-II Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Population size: {config.population_size}\n")
        f.write(f"Generations: {config.n_generations}\n")
        f.write(f"Block size: {config.block_size}\n")
        f.write(f"Block grid: {config.block_grid_h}x{config.block_grid_w}\n\n")

        f.write("Pareto Front Analysis:\n")
        f.write("-" * 30 + "\n")
        for key, value in analysis.items():
            f.write(f"{key}: {value}\n")

        f.write("\nBest solutions by priority:\n")
        f.write("-" * 30 + "\n")
        for priority in ['imperceptibility', 'payload', 'security', 'balanced']:
            best = select_best_solution(population, pareto_indices, priority=priority)
            f.write(f"{priority}: alpha={best.alpha:.4f}, density={best.mask_genes.mean():.4f}\n")

    logging.info(f"\nResults saved to {results_file}")

    # Save best masks
    for priority in ['imperceptibility', 'payload', 'security', 'balanced']:
        best = select_best_solution(population, pareto_indices, priority=priority)
        mask_file = os.path.join(args.output_dir, f'mask_{priority}.npy')
        np.save(mask_file, best.mask_genes)
        # save corresponding alpha
        alpha_file = os.path.join(args.output_dir, f'alpha_{priority}.txt')
        with open(alpha_file, 'w') as af:
            af.write(str(float(best.alpha)))
        logging.info(f"Saved mask to {mask_file}")

    # Save pareto front CSV and summary JSON
    try:
        import csv, json
        pareto_csv = os.path.join(args.output_dir, 'pareto_front.csv')
        with open(pareto_csv, 'w', newline='') as pcsv:
            writer = csv.writer(pcsv)
            writer.writerow(['index', 'alpha', 'mask_density', 'f1', 'f2', 'f3'])
            for i, idx in enumerate(pareto_indices):
                sol = population[idx]
                writer.writerow([i, sol.alpha, sol.mask_genes.mean(), sol.fitness[0], sol.fitness[1], sol.fitness[2]])

        summary = {
            'nsga2_config': vars(config),
            'pareto_size': len(pareto_indices),
            'selected_priorities': {}
        }
        for priority in ['imperceptibility', 'payload', 'security', 'balanced']:
            best = select_best_solution(population, pareto_indices, priority=priority)
            summary['selected_priorities'][priority] = {
                'alpha': float(best.alpha),
                'mask_density': float(best.mask_genes.mean())
            }

        summary_file = os.path.join(args.output_dir, 'nsga2_summary.json')
        with open(summary_file, 'w') as sf:
            json.dump(summary, sf, indent=2)

        logging.info(f"Saved pareto front CSV to {pareto_csv} and summary to {summary_file}")
    except Exception as ex:
        logging.warning(f"Failed to save pareto CSV or summary JSON: {ex}")

    logging.info("\nOptimization complete!")


if __name__ == '__main__':
    main()