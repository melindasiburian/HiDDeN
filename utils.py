import numpy as np
import os
import re
import csv
import time
import pickle
import logging

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    # pass nrow as a keyword argument so an integer isn't mistaken for the
    # PIL 'format' parameter in some torchvision versions where the
    # signature is (tensor, fp, format=None, **kwargs)
    torchvision.utils.save_image(stacked_images, filename, nrow=original_images.shape[0], normalize=False)


def save_visualizations(original_images, encoded_images, noised_images=None, mask=None,
                        epoch: int = 0, folder: str = '.', resize_to=None, images_to_save: int = 8,
                        amp_factor: float = 5.0, active_threshold: float = 0.02):
    """
    Save visualization figures for a batch: original vs encoded, residual heatmap, mask heatmap (optional),
    and histograms of residuals. Creates one combined figure per image.

    Args:
        original_images: tensor [B, C, H, W] in range [-1, 1]
        encoded_images: tensor [B, C, H, W] in range [-1, 1]
        noised_images: optional tensor [B, C, H, W]
        mask: optional tensor [B, 1, H, W] or [B, H, W]
        epoch: epoch number used in filenames
        folder: output folder
        resize_to: tuple or None, if provided images will be resized for display
        images_to_save: how many images from batch to save
        amp_factor: factor to amplify residual for visualization
        active_threshold: threshold on normalized residual magnitude for active area mask
    """
    os.makedirs(folder, exist_ok=True)

    B = min(original_images.shape[0], images_to_save)
    # move to cpu and float
    orig = original_images[:B].cpu().float()
    enc = encoded_images[:B].cpu().float()
    if noised_images is not None:
        noised = noised_images[:B].cpu().float()
    else:
        noised = None

    for i in range(B):
        o = orig[i]
        e = enc[i]

        if resize_to is not None:
            o_disp = F.interpolate(o.unsqueeze(0), size=resize_to).squeeze(0)
            e_disp = F.interpolate(e.unsqueeze(0), size=resize_to).squeeze(0)
            if noised is not None:
                n_disp = F.interpolate(noised[i].unsqueeze(0), size=resize_to).squeeze(0)
        else:
            o_disp = o
            e_disp = e
            if noised is not None:
                n_disp = noised[i]

        # images for display in uint8
        o_img = tensor_to_image(o_disp.unsqueeze(0))[0]
        e_img = tensor_to_image(e_disp.unsqueeze(0))[0]

        # residual (float)
        residual = e_disp - o_disp
        # per-pixel magnitude (average over channels)
        residual_mag = residual.abs().mean(dim=0).numpy()

        # normalize residual magnitude for visualization
        max_val = residual_mag.max() if residual_mag.max() > 0 else 1.0
        residual_norm = residual_mag / (max_val + 1e-12)

        # active area mask
        active_mask = (residual_norm > active_threshold).astype(float)

        # amplified residual for display (clipped)
        amplified = (residual_norm * amp_factor).clip(0, 1)

        # prepare histogram data (signed residual values across channels)
        residual_vals = residual.view(-1).numpy()

        # mask heatmap if provided
        mask_heat = None
        if mask is not None:
            m = mask[:B].cpu()
            mm = m[i]
            if mm.dim() == 3:
                mm = mm.squeeze(0)
            if resize_to is not None:
                mm = F.interpolate(mm.unsqueeze(0).unsqueeze(0), size=resize_to).squeeze().numpy()
            else:
                mm = mm.numpy()
            mask_heat = mm

        # build figure with subplots
        cols = 5 if mask_heat is not None else 4
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

        ax = axes[0]
        ax.imshow(o_img)
        ax.set_title('Original')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(e_img)
        ax.set_title('Encoded')
        ax.axis('off')

        ax = axes[2]
        im = ax.imshow(residual_norm, cmap='hot')
        ax.set_title('Residual heatmap (norm)')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[3]
        ax.imshow(amplified, cmap='inferno')
        ax.set_title(f'Amplified x{amp_factor}')
        ax.axis('off')

        if mask_heat is not None:
            ax = axes[4]
            ax.imshow(mask_heat, cmap='Blues')
            ax.set_title('Mask heatmap')
            ax.axis('off')

        # create a separate figure for histogram to keep layout clean
        hist_fig, hist_ax = plt.subplots(1, 1, figsize=(6, 4))
        hist_ax.hist(residual_vals, bins=100, color='gray')
        hist_ax.set_title('Residual distribution (signed)')
        hist_ax.set_xlabel('Residual value')
        hist_ax.set_ylabel('Count')

        # save both figures
        fname_base = os.path.join(folder, f'epoch-{epoch:04d}_img-{i}')
        fig.savefig(fname_base + '_summary.png', bbox_inches='tight')
        hist_fig.savefig(fname_base + '_hist.png', bbox_inches='tight')
        plt.close(fig)
        plt.close(hist_fig)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> tuple[TrainingOptions, HiDDenConfiguration, dict]:
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    # Base folder name uses a timestamp. If a folder with the same name already exists
    # (e.g., a previous run started at the same second), append a numeric suffix to
    # pick a unique folder name rather than raising FileExistsError.
    base_name = f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}'
    this_run_folder = os.path.join(runs_folder, base_name)

    suffix = 1
    while os.path.exists(this_run_folder):
        this_run_folder = os.path.join(runs_folder, f"{base_name}-{suffix}")
        suffix += 1

    # Create the run folder and standard subfolders. Use exist_ok=True to be robust
    # (in case of race conditions or concurrent processes), but we've already chosen
    # a unique folder name so these should normally be created fresh.
    os.makedirs(this_run_folder, exist_ok=True)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(this_run_folder, 'images'), exist_ok=True)

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    # On Windows a different process (e.g. a spreadsheet app) can hold an
    # exclusive lock on the CSV. Instead of failing immediately, retry a few
    # times with exponential backoff so transient locks don't crash training.
    max_attempts = 5
    base_delay = 0.5
    row_header = None
    row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
        '{:.0f}'.format(duration)]

    for attempt in range(1, max_attempts + 1):
        try:
            # Open, write and close quickly so we don't hold the file longer than needed.
            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if epoch == 1:
                    row_header = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
                    writer.writerow(row_header)
                writer.writerow(row_to_write)
            break
        except PermissionError:
            if attempt == max_attempts:
                # Final attempt failed: instead of raising (which would stop training),
                # fall back to appending into a separate pending file so data isn't lost
                # and training can continue. The pending file can be merged later
                # when the CSV is available.
                pending_file = file_name + '.pending'
                try:
                    with open(pending_file, 'a', newline='') as pfile:
                        pwriter = csv.writer(pfile)
                        # write header if this is the first epoch write
                        if epoch == 1:
                            pwriter.writerow(['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration'])
                        pwriter.writerow(row_to_write)
                    logging.warning(f"Could not write to {file_name} due to file lock; appended to {pending_file} instead.")
                    break
                except Exception:
                    # If even the pending file cannot be written (very rare), re-raise
                    raise
            # Wait a bit and retry. Use exponential backoff to reduce contention.
            time.sleep(base_delay * (2 ** (attempt - 1)))