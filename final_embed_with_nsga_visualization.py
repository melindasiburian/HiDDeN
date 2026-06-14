"""
Final embedding script that applies a mask and alpha found by NSGA-II to encode a message using a trained HiDDeN encoder.

Usage example (as requested):
python final_embed_with_nsga_visualization.py \
  --options-file ./runs/NAMA_RUN/options-and-config.pickle \
  --checkpoint-file ./runs/NAMA_RUN/checkpoints/NAMA_CHECKPOINT.pyt \
  --source-image ./data/val/cover/sample.jpg \
  --mask-file ./nsga2_results/mask_balanced.npy \
  --alpha-file ./nsga2_results/alpha_balanced.txt \
  --output-dir ./final_embedding_results \
  --message-length 30 \
  --amplification 10

If --alpha-file is not provided you may pass --alpha 0.8

Outputs (saved in output-dir):
- cover.png
- encoded.png
- difference.png
- residual_heatmap.png
- mask_heatmap.png
- masked_residual_heatmap.png
- residual_amplified.png
- overlay_mask_on_cover.png
- residual_histogram.png
- difference_histogram.png
- metrics.json
- original_message.txt
- decoded_message.txt

This script does NOT retrain the model. It runs the encoder in eval mode with torch.no_grad().
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import utils
from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden
from noise_layers.noiser import Noiser


def load_options_and_models(options_file: str, checkpoint_file: str, device: torch.device):
    if not os.path.exists(options_file):
        raise FileNotFoundError(f"Options file not found: {options_file}")
    train_options, hidden_config, noise_config = utils.load_options(options_file)

    # create noiser and models
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # load state dicts
    model.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    model.discriminator.load_state_dict(checkpoint['discrim-model'])

    # freeze and eval
    model.encoder_decoder.eval()
    model.discriminator.eval()

    for p in model.encoder_decoder.parameters():
        p.requires_grad = False
    for p in model.discriminator.parameters():
        p.requires_grad = False

    return train_options, hidden_config, model


def load_image_as_tensor(path: str, H: int, W: int, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source image not found: {path}")
    img = Image.open(path).convert('RGB')
    img = TF.center_crop(img, (H, W))
    t = TF.to_tensor(img).to(device)
    # transform [0,1] -> [-1,1]
    t = t * 2 - 1
    t = t.unsqueeze(0)  # [1, C, H, W]
    return t


def save_image_uint8(tensor, path: str):
    # tensor in range [-1,1] or [0,1]
    img = tensor.clone().detach()
    if img.min() < -0.5:
        img = (img + 1) / 2
    img = img.clamp(0, 1)
    img = TF.to_pil_image(img.squeeze(0))
    img.save(path)


def ensure_mask_tensor(mask_path: str, H: int, W: int, device: torch.device):
    # supports mask in .npy or raw array
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask_np = np.load(mask_path)
    # normalize to binary 0/1
    mask_np = (mask_np > 0.5).astype(np.float32)

    # mask shape cases: [H, W], [1, 1, H, W], [B, 1, H, W]
    if mask_np.ndim == 2:
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    elif mask_np.ndim == 3:
        # could be [1, H, W] or [B, H, W]
        if mask_np.shape[0] == 1:
            mask_t = torch.from_numpy(mask_np).unsqueeze(1).to(device)
        else:
            mask_t = torch.from_numpy(mask_np).unsqueeze(1).to(device)
    elif mask_np.ndim == 4:
        mask_t = torch.from_numpy(mask_np).to(device)
    else:
        raise ValueError(f"Unsupported mask ndarray shape: {mask_np.shape}")

    # if mask is block-level (values are 0/1 but small dims), upsample
    if mask_t.shape[-2] != H or mask_t.shape[-1] != W:
        mask_t = F.interpolate(mask_t.float(), size=(H, W), mode='nearest')

    return mask_t.float()


def read_alpha(alpha_file: str, alpha_arg: float = None):
    if alpha_arg is not None:
        return float(alpha_arg)
    if alpha_file is None:
        raise ValueError('Alpha file or --alpha must be provided')
    if not os.path.exists(alpha_file):
        raise FileNotFoundError(f"Alpha file not found: {alpha_file}")
    try:
        with open(alpha_file, 'r') as f:
            v = float(f.read().strip())
    except Exception:
        # try json
        with open(alpha_file, 'r') as f:
            j = json.load(f)
            v = float(j.get('alpha', 1.0))
    return v


def main():
    parser = argparse.ArgumentParser(description='Final embedding with NSGA-II mask/alpha and visualization')
    parser.add_argument('--options-file', required=True)
    parser.add_argument('--checkpoint-file', required=True)
    parser.add_argument('--source-image', required=True)
    parser.add_argument('--mask-file', required=True)
    parser.add_argument('--alpha-file', required=False)
    parser.add_argument('--alpha', type=float, required=False)
    parser.add_argument('--message-length', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--amplification', type=float, default=5.0)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    train_options, hidden_config, model = load_options_and_models(args.options_file, args.checkpoint_file, device)

    # image size
    H, W = hidden_config.H, hidden_config.W

    cover = load_image_as_tensor(args.source_image, H, W, device)

    # load mask
    mask = ensure_mask_tensor(args.mask_file, H, W, device)
    # broadcast mask to batch
    if mask.shape[0] == 1:
        mask = mask.expand(1, -1, -1, -1)

    # read alpha
    alpha_val = read_alpha(args.alpha_file, args.alpha)
    alpha = torch.tensor(alpha_val, dtype=torch.float32, device=device)

    # prepare message (random binary)
    message = torch.randint(0, 2, (1, args.message_length), dtype=torch.float32, device=device)

    # Run encoder to get residual; EncoderDecoder.forward returns encoded_image, noised_image, decoded_message
    model.encoder_decoder.eval()
    with torch.no_grad():
        encoded_image, noised_image, decoded_message = model.encoder_decoder(cover, message, mask=mask, alpha=alpha)

    # Residual calculation: residual = encoder output that was added to image
    # The Encoder in this repo typically returns encoded_image = image + alpha * mask * residual
    # so residual = (encoded_image - cover) / (alpha * mask + eps)
    eps = 1e-8
    denom = (alpha * mask).clone()
    denom[denom == 0] = eps
    residual_est = (encoded_image - cover) / denom

    # Final encoded image (apply same formula to be explicit)
    final_encoded = cover + alpha * mask * residual_est

    # Run decoder on final encoded (in case we want to evaluate) -- ensure eval & no_grad
    with torch.no_grad():
        # If encoder_decoder expects encoder->noiser->decoder pipeline, we can call decoder directly
        # but using encoder_decoder validate path is safer
        # We'll call encoder_decoder in eval mode but bypassing training noise
        # For evaluation, call encoder_decoder with already-encoded image by replacing the encoder output
        # Simpler: call model.encoder_decoder.encoder to get residual and then pass encoded into decoder
        # However EncoderDecoder doesn't expose a direct 'decode' interface here; we'll use existing decoder
        # In the repository EncoderDecoder.forward returns (encoded, noised, decoded)
        # So we can run decoder on noised version of final_encoded via noiser
        encoded_for_decoder = final_encoded
        noised_and_cover = model.encoder_decoder.noiser([encoded_for_decoder, cover])
        noised_img = noised_and_cover[0]
        decoded = model.encoder_decoder.decoder(noised_img)

        # Discriminator score
        discrim_logits = model.discriminator(final_encoded)
        discrim_probs = torch.sigmoid(discrim_logits).squeeze().cpu().numpy()

    # Metrics
    psnr = float(utils.calculate_psnr(cover, final_encoded, data_range=2.0).mean().item())
    ssim = float(utils.calculate_ssim(cover, final_encoded, data_range=2.0).mean().item())
    ber = float(utils.calculate_ber(decoded, message).mean().item())
    bpp = float(utils.calculate_bpp(args.message_length, H, W))
    security = float(utils.calculate_security_score(model.discriminator, final_encoded, cover).mean().item())

    metrics = {
        'psnr': psnr,
        'ssim': ssim,
        'ber': ber,
        'bpp': bpp,
        'security': security,
        'alpha': float(alpha_val),
        'mask_density': float((mask > 0.5).float().mean().item()),
        'message_length': args.message_length,
        'checkpoint': args.checkpoint_file,
        'mask_file': args.mask_file
    }

    # Save images and visualizations
    save_image_uint8(((cover + 1) / 2).cpu(), os.path.join(args.output_dir, 'cover.png'))
    save_image_uint8(((final_encoded + 1) / 2).cpu(), os.path.join(args.output_dir, 'encoded.png'))

    # difference
    diff = (final_encoded - cover).abs()
    save_image_uint8(diff.cpu(), os.path.join(args.output_dir, 'difference.png'))

    # residual heatmap and masked residual
    residual_mag = residual_est.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    masked_residual = (alpha * mask * residual_est).abs().mean(dim=1, keepdim=True)

    save_image_uint8(residual_mag.cpu(), os.path.join(args.output_dir, 'residual_heatmap.png'))
    save_image_uint8(masked_residual.cpu(), os.path.join(args.output_dir, 'masked_residual_heatmap.png'))

    # amplified residual visualization
    residual_amp = (diff.abs() * args.amplification).clamp(0, 1)
    save_image_uint8(residual_amp.cpu(), os.path.join(args.output_dir, 'residual_amplified.png'))

    # mask heatmap
    mask_vis = mask.float().mean(dim=1, keepdim=True)  # [B,1,H,W]
    save_image_uint8(mask_vis.cpu(), os.path.join(args.output_dir, 'mask_heatmap.png'))

    # overlay mask on cover
    # convert cover to uint8 image for overlay
    cover_img = TF.to_pil_image(((cover + 1) / 2).clamp(0,1).squeeze(0).cpu())
    mask_img = TF.to_pil_image(mask_vis.squeeze(0).repeat(3,1,1).cpu())
    overlay = Image.blend(cover_img.convert('RGBA'), mask_img.convert('RGBA'), alpha=0.4)
    overlay.save(os.path.join(args.output_dir, 'overlay_mask_on_cover.png'))

    # histograms
    residual_vals = (residual_est.view(-1).cpu().numpy())
    diff_vals = (diff.view(-1).cpu().numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.hist(residual_vals, bins=200)
    plt.title('Residual histogram')
    plt.savefig(os.path.join(args.output_dir, 'residual_histogram.png'))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(diff_vals, bins=200)
    plt.title('Difference histogram')
    plt.savefig(os.path.join(args.output_dir, 'difference_histogram.png'))
    plt.close()

    # Save messages
    np.savetxt(os.path.join(args.output_dir, 'original_message.txt'), message.cpu().numpy().astype(int), fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'decoded_message.txt'), decoded.detach().cpu().numpy().round().astype(int), fmt='%d')

    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
        json.dump(metrics, mf, indent=2)

    print('Saved final embedding outputs to', args.output_dir)


if __name__ == '__main__':
    main()
