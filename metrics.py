"""
Metrics for evaluating steganography performance.

This module provides functions to calculate:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- BPP (Bits Per Pixel)
- BER (Bit Error Rate)
- Security score (from discriminator/steganalyzer)

All functions support batched PyTorch tensors.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
from typing import Optional, Any


def calculate_psnr(cover: Any, encoded: Any, data_range: float = 2.0) -> Any:
    """
    Calculate PSNR between cover and encoded images.

    Args:
        cover: Cover image tensor [B, C, H, W], range [-1, 1] or [0, 1]
        encoded: Encoded/stego image tensor [B, C, H, W], range [-1, 1] or [0, 1]
        data_range: The dynamic range of the images (2.0 for [-1,1], 1.0 for [0,1])

    Returns:
        PSNR values per image [B]
    """
    batch_size = cover.shape[0]
    mse = F.mse_loss(cover, encoded, reduction='none').view(batch_size, -1).mean(dim=1)

    # Avoid division by zero
    mse = torch.where(mse == 0, torch.ones_like(mse) * 1e-10, mse)

    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr


def calculate_ssim(cover: Any, encoded: Any, window_size: int = 11,
                    data_range: float = 2.0) -> Any:
    """
    Calculate SSIM between cover and encoded images.

    Uses a simplified SSIM implementation compatible with batched tensors.

    Args:
        cover: Cover image tensor [B, C, H, W], range [-1, 1] or [0, 1]
        encoded: Encoded/stego image tensor [B, C, H, W], range [-1, 1] or [0, 1]
        window_size: Size of Gaussian window
        data_range: The dynamic range of the images

    Returns:
        SSIM values per image [B]
    """
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
        for x in range(window_size)
    ], dtype=cover.dtype, device=cover.device)
    gauss = gauss / gauss.sum()

    # Create 2D window
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]

    # Expand window to match input channels
    num_channels = cover.shape[1]
    window = window.expand(num_channels, 1, -1, -1)  # [C, 1, window_size, window_size]

    # Compute means
    mu_cover = F.conv2d(cover, window, padding=window_size // 2, groups=num_channels)
    mu_encoded = F.conv2d(encoded, window, padding=window_size // 2, groups=num_channels)

    mu_cover_sq = mu_cover ** 2
    mu_encoded_sq = mu_encoded ** 2
    mu_cover_encoded = mu_cover * mu_encoded

    # Compute variances and covariance
    sigma_cover_sq = F.conv2d(cover ** 2, window, padding=window_size // 2, groups=num_channels) - mu_cover_sq
    sigma_encoded_sq = F.conv2d(encoded ** 2, window, padding=window_size // 2, groups=num_channels) - mu_encoded_sq
    sigma_cover_encoded = F.conv2d(cover * encoded, window, padding=window_size // 2, groups=num_channels) - mu_cover_encoded

    # SSIM formula
    numerator = (2 * mu_cover_encoded + C1) * (2 * sigma_cover_encoded + C2)
    denominator = (mu_cover_sq + mu_encoded_sq + C1) * (sigma_cover_sq + sigma_encoded_sq + C2)

    ssim = numerator / (denominator + 1e-10)

    # Average over channels and spatial dimensions
    ssim = ssim.mean(dim=[1, 2, 3])

    return ssim


def calculate_bpp(message_length: int, H: int, W: int) -> float:
    """
    Calculate theoretical bits per pixel (BPP).

    Args:
        message_length: Length of message in bits
        H: Image height in pixels
        W: Image width in pixels

    Returns:
        BPP value
    """
    total_pixels = H * W
    bpp = message_length / total_pixels
    return bpp


def calculate_ber(decoded_message: Any, original_message: Any) -> Any:
    """
    Calculate Bit Error Rate between decoded and original messages.

    Args:
        decoded_message: Decoded message tensor [B, message_length], values in [0, 1] or [0, 1]
        original_message: Original message tensor [B, message_length], binary values

    Returns:
        BER values per image [B]
    """
    # Round decoded message to get binary predictions
    decoded_binary = torch.round(torch.clamp(decoded_message, 0, 1))

    # Count bit errors
    bit_errors = torch.sum(torch.abs(decoded_binary - original_message), dim=1)

    # BER = bit_errors / message_length
    message_length = original_message.shape[1]
    ber = bit_errors.float() / message_length

    return ber


def calculate_security_score(discriminator: Any, encoded_image: Any,
                              cover_image: Optional[Any] = None) -> Any:
    """
    Calculate security score based on discriminator output.

    The discriminator tries to distinguish cover from stego images.
    A good steganography scheme should fool the discriminator (low detection probability).

    Args:
        discriminator: Trained discriminator model
        encoded_image: Stego/encoded image tensor [B, C, H, W]
        cover_image: Optional cover image for comparison [B, C, H, W]

    Returns:
        Detection probability (lower is better for security) [B]
    """
    discriminator.eval()

    with torch.no_grad():
        # Get discriminator output for encoded (stego) images
        stego_logits = discriminator(encoded_image)
        stego_probs = torch.sigmoid(stego_logits)

        # If cover images provided, also get cover probabilities
        if cover_image is not None:
            cover_logits = discriminator(cover_image)
            cover_probs = torch.sigmoid(cover_logits)
            # Average detection probability
            detection_prob = (stego_probs + (1 - cover_probs)) / 2
        else:
            # Just use stego detection probability
            # Assuming discriminator outputs 1 for cover, 0 for stego
            # So detection probability = probability of being cover
            detection_prob = stego_probs

    return detection_prob.squeeze(-1)  # [B]


def calculate_security_score_batch(discriminator: Any, images: Any,
                                   labels: Any) -> float:
    """
    Calculate security score over a batch with known labels.

    Args:
        discriminator: Trained discriminator model
        images: Image tensor [B, C, H, W]
        labels: Labels [B, 1] (1=cover, 0=stego)

    Returns:
        Average detection accuracy (lower is better for security)
    """
    discriminator.eval()

    with torch.no_grad():
        logits = discriminator(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == labels).float().mean()

    return accuracy.item()


def evaluate_all_metrics(cover: Any, encoded: Any,
                        decoded_message: Any, original_message: Any,
                        discriminator: Optional[Any],
                        message_length: int, data_range: float = 2.0) -> dict:
    """
    Calculate all evaluation metrics in one function.

    Args:
        cover: Cover image tensor [B, C, H, W]
        encoded: Encoded/stego image tensor [B, C, H, W]
        decoded_message: Decoded message tensor [B, message_length]
        original_message: Original message tensor [B, message_length]
        discriminator: Optional discriminator for security scoring
        message_length: Length of message in bits
        data_range: Dynamic range of images

    Returns:
        Dictionary with all metrics (values are tensors [B] or scalar)
    """
    metrics = {}

    # PSNR
    metrics['psnr'] = calculate_psnr(cover, encoded, data_range)

    # SSIM
    metrics['ssim'] = calculate_ssim(cover, encoded, data_range=data_range)

    # BPP (same for entire batch)
    B, C, H, W = cover.shape
    metrics['bpp'] = torch.tensor([calculate_bpp(message_length, H, W)] * B)

    # BER
    metrics['ber'] = calculate_ber(decoded_message, original_message)

    # Security score
    if discriminator is not None:
        metrics['security'] = calculate_security_score(discriminator, encoded, cover)
    else:
        metrics['security'] = torch.zeros(B)

    return metrics


# ==============================================================================
# UTILITY FUNCTIONS FOR MASK OPERATIONS
# ==============================================================================

def create_block_mask(block_size: int, H: int, W: int, density: float,
                      device: Any = torch.device('cpu')) -> Any:
    """
    Create a block-level binary mask.

    Args:
        block_size: Size of each block (block_size x block_size)
        H: Image height
        W: Image width
        density: Fraction of blocks that are 1 (embedding locations)
        device: Device to create tensor on

    Returns:
        Binary mask tensor [1, 1, H, W]
    """
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size

    # Create random block mask
    n_blocks = n_blocks_h * n_blocks_w
    n_ones = int(n_blocks * density)

    block_mask = torch.zeros(n_blocks, device=device)
    block_mask[:n_ones] = 1
    block_mask = block_mask[torch.randperm(n_blocks, device=device)]

    # Reshape to block grid
    block_mask = block_mask.view(n_blocks_h, n_blocks_w)

    # Upsample to full resolution using nearest neighbor
    mask = F.interpolate(
        block_mask.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    )

    return mask


def upsample_block_mask(block_genes: Any, target_h: int, target_w: int) -> Any:
    """
    Upsample block-level genes to full resolution mask.

    Args:
        block_genes: Block-level genes [B, 1, block_h, block_w] or [block_h, block_w]
        target_h: Target height
        target_w: Target width

    Returns:
        Upsampled mask [B, 1, target_h, target_w] or [1, 1, target_h, target_w]
    """
    if block_genes.dim() == 2:
        block_genes = block_genes.unsqueeze(0).unsqueeze(0)
    elif block_genes.dim() == 3:
        block_genes = block_genes.unsqueeze(1)

    # Upsample using nearest neighbor
    mask = F.interpolate(
        block_genes.float(),
        size=(target_h, target_w),
        mode='nearest'
    )

    return mask


def calculate_mask_density(mask: Any) -> float:
    """
    Calculate the density (fraction of 1s) in a binary mask.

    Args:
        mask: Binary mask tensor

    Returns:
        Density value between 0 and 1
    """
    if mask.dim() > 1:
        mask = mask.view(-1)
    return (mask > 0.5).float().mean().item()


def calculate_payload(mask: Any, message_length: int) -> float:
    """
    Calculate effective payload based on embedding mask and message length.

    Args:
        mask: Binary mask tensor [B, 1, H, W] or [B, H, W]
        message_length: Length of the watermark message in bits

    Returns:
        Effective payload (bits per pixel times mask density ratio)
    """
    mask_density = calculate_mask_density(mask)
    # Payload is proportional to mask density times message length
    payload = mask_density * message_length
    return payload