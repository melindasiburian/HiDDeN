import pickle, pprint, os, torch, sys
from pathlib import Path

# Ensure project root is on sys.path so local imports (options, utils, model, etc.) work
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
# Load configs
p = Path(r'runs/hidden_baseline 2026.05.09--08-59-13/options-and-config.pickle')
with open(p,'rb') as f:
    train_options = pickle.load(f)
    noise_config = pickle.load(f)
    hidden_config = pickle.load(f)

# find last checkpoint
ckpt_dir = Path(r'runs/hidden_baseline 2026.05.09--08-59-13/checkpoints')
ckpts = sorted(ckpt_dir.glob('*.pyt'))
if not ckpts:
    print('No checkpoints found in', ckpt_dir)
    raise SystemExit(1)
last_ckpt = str(ckpts[-1])
print('Using checkpoint:', last_ckpt)

# import project modules
import utils, metrics
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from options import HiDDenConfiguration, TrainingOptions

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build Noiser from loaded noise_config: Noiser expects (noise_layers, device)
if isinstance(noise_config, Noiser):
    noiser = noise_config
else:
    noiser = Noiser(noise_config, device)

# build Hidden
hidden = Hidden(hidden_config, device, noiser, tb_logger=None)
# load checkpoint into model
ckpt = torch.load(last_ckpt, map_location=device)
try:
    hidden.encoder_decoder.load_state_dict(ckpt['enc-dec-model'])
    # optimizer states may not load cleanly; do not require them
    try:
        hidden.optimizer_enc_dec.load_state_dict(ckpt['enc-dec-optim'])
        hidden.discriminator.load_state_dict(ckpt['discrim-model'])
        hidden.optimizer_discrim.load_state_dict(ckpt['discrim-optim'])
    except Exception:
        pass
except Exception as e:
    print('Warning loading model state:', e)

# get a validation batch
train_loader, val_loader = utils.get_data_loaders(hidden_config, train_options)
print('Validation batch size:', train_options.batch_size)
try:
    batch = next(iter(val_loader))
except Exception as e:
    print('Failed to get validation batch:', e)
    raise
images, _ = batch
# create random messages
messages = torch.randint(0,2,(images.shape[0], hidden_config.message_length)).float()

images = images.to(device)
messages = messages.to(device)

# run through model in eval
hidden.encoder_decoder.eval()
hidden.discriminator.eval()
with torch.no_grad():
    encoded, noised, decoded = hidden.encoder_decoder(images, messages)

# ensure decoded in [0,1]
if isinstance(decoded, torch.Tensor):
    decoded_sig = torch.sigmoid(decoded) if decoded.max()>1.5 or decoded.min()< -0.5 else decoded
else:
    decoded_sig = torch.tensor(decoded)

# compute metrics
psnr = metrics.calculate_psnr(images, encoded, data_range=2.0)
ssim = metrics.calculate_ssim(images, encoded, data_range=2.0)
ber = metrics.calculate_ber(decoded_sig, messages)

print('\nMetrics (per-image) sample:')
print('PSNR (first 8):', psnr[:8].cpu().numpy())
print('SSIM (first 8):', ssim[:8].cpu().numpy())
print('BER (first 8):', ber[:8].cpu().numpy())
print('\nMeans:')
print('PSNR mean:', float(psnr.mean().item()))
print('SSIM mean:', float(ssim.mean().item()))
print('BER mean:', float(ber.mean().item()))
