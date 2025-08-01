import utils
from torch.utils.data import DataLoader, random_split
from models.vae import VAE
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

# 1. 引入 SummaryWriter
from torch.utils.tensorboard import SummaryWriter

data_dir = "../bev_tensors"

parser = argparse.ArgumentParser()
"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_updates", type=int, default=10000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)
parser.add_argument("--tb_subdir", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/model' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""
full_dataset = utils.TensorDataset(data_dir)
val_ratio = 0.1
val_size = int(val_ratio * len(full_dataset))
train_size = len(full_dataset) - val_size

training_data, validation_data = random_split(full_dataset, [train_size, val_size])
training_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

model = VAE(args.n_hiddens, args.n_residual_hiddens,
            args.n_residual_layers, args.latent_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
}


def vae_loss(x_hat, x, mu, logvar, beta=10.0):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def evaluate(model, val_loader):
    model.eval()
    total_recon, total_kl, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_samples += x.size(0)
    model.train()
    avg_recon = total_recon / n_samples
    avg_kl = total_kl / n_samples
    return avg_recon, avg_kl

def train():
    update_count = 0
    if args.tb_subdir:
        writer = SummaryWriter(log_dir=f"runs/{args.tb_subdir}")
    else:
        writer = SummaryWriter()

    pbar = tqdm(total=args.n_updates, desc="Training", unit="it")


    while update_count < args.n_updates:
        for x in training_loader:
            x = x.to(device, non_blocking=True)
            torch.backends.cudnn.benchmark = True

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.item())
            results["loss_vals"].append(loss.item())
            results["n_updates"] = update_count

            writer.add_scalar("Train/TotalLoss", loss.item(), update_count)
            writer.add_scalar("Train/ReconstructionLoss", recon_loss.item(), update_count)
            writer.add_scalar("Train/KLDiv", kl_loss.item(), update_count)

            pbar.update(1)
            pbar.set_postfix({
                "recon": f"{recon_loss.item():.2f}",
                "kl":    f"{kl_loss.item():.2f}",
                "total": f"{loss.item():.2f}"
            })

            if update_count % args.log_interval == 0:
                if args.save:
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename)
                print(
                    f'Update #{update_count:05d} | Recon Error: {recon_loss.item():.6f} '
                    f'| KL: {kl_loss.item():.6f} | Total Loss: {loss.item():.6f}'
                )

            update_count += 1

            if update_count % 500 == 0:
                torch.save(model.encoder.state_dict(), f"./results/encoder/vae_encoder_step{update_count}.pth")
                torch.save(model.decoder.state_dict(), f"./results/decoder/vae_decoder_step{update_count}.pth")

            if update_count % (args.log_interval * 5) == 0:
                avg_recon_val, avg_kl_val = evaluate(model, validation_loader)
                writer.add_scalar("Val/ReconstructionLoss", avg_recon_val, update_count)
                writer.add_scalar("Val/KLDiv", avg_kl_val, update_count)
                writer.add_scalar("Val/TotalLoss", avg_recon_val + avg_kl_val, update_count)
                print(f'验证集 | Recon Error: {avg_recon_val:.4f} | KL: {avg_kl_val:.4f} | Total: {avg_recon_val+avg_kl_val:.4f}')

            if update_count >= args.n_updates:
                torch.save(model.encoder.state_dict(), "./results/encoder/vae_encoder_final.pth")
                torch.save(model.decoder.state_dict(), "./results/decoder/vae_decoder_final.pth")

                break

    pbar.close()
    writer.close()

if __name__ == "__main__":
    train()
