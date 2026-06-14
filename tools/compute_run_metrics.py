import pickle, pprint, torch, os
from options import HiDDenConfiguration, TrainingOptions
from utils import get_data_loaders, load_last_checkpoint
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from metrics import calculate_psnr, calculate_ssim, calculate_ber

p = r'runs\\hidden_baseline 2026.05.09--08-59-13\\options-and-config.pickle'
with open(p,'rb') as f:
    train_options = pickle.load(f)
    noise_config = pickle.load(f)
    hidden_config = pickle.load(f)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Build noiser
noiser = Noiser(noise_config, device)

# Build Hidden
hidden = Hidden(hidden_config, device, noiser, tb_logger=None)

# load last checkpoint
ckpt_folder = os.path.join('runs','hidden_baseline 2026.05.09--08-59-13','checkpoints')
ckpt, ckpt_path = load_last_checkpoint(ckpt_folder)
hidden.encoder_decoder.load_state_dict(ckpt['enc-dec-model'])
print('Loaded checkpoint:', ckpt_path)

# get dataloaders
train_loader, val_loader = get_data_loaders(hidden_config, train_options)

# take one batch from validation
batch = next(iter(val_loader))
images, _ = batch
# prepare message (random same as training)
message = torch.Tensor(torch.randint(0,2,(images.shape[0], hidden_config.message_length))).to(device)
images = images.to(device)

hidden.encoder_decoder.eval()
with torch.no_grad():
    encoded, noised, decoded = hidden.encoder_decoder(images.to(device), message.to(device))

# compute metrics (PSNR/SSIM expect range [-1,1] data_range=2.0)
psnr = calculate_psnr(images, encoded, data_range=2.0)
ssim = calculate_ssim(images, encoded, data_range=2.0)
ber = calculate_ber(decoded, message)

print('PSNR mean:', float(psnr.mean().item()))
print('SSIM mean:', float(ssim.mean().item()))
print('BER mean:', float(ber.mean().item()))
