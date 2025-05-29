import utils
from torch.utils.data import DataLoader, random_split
from models.vae import VAE
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

data_dir = "../bev_tensors"

parser = argparse.ArgumentParser()
"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=10000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)
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
training_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                               pin_memory=True)

model = VAE(args.n_hiddens, args.n_residual_hiddens,
            args.n_residual_layers, args.latent_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
}


def vae_loss(x_hat, x, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


def evaluate(model, val_loader):
    model.eval()
    recon_losses, kl_losses = [], []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
    model.train()


def train():
    update_count = 0
    pbar = tqdm(total=args.n_updates, desc="Training", unit="it")  # 创建总进度条

    while update_count < args.n_updates:
        for x in training_loader:
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)

            loss.backward()
            optimizer.step()

            # 记录
            results["recon_errors"].append(recon_loss.item())
            results["loss_vals"].append(loss.item())
            results["n_updates"] = update_count

            # 更新进度条并加上当前 loss 信息
            pbar.update(1)
            pbar.set_postfix({
                "recon": f"{recon_loss.item():.2f}",
                "kl":    f"{kl_loss.item():.2f}",
                "total": f"{loss.item():.2f}"
            })

            # 按原逻辑定期保存 & 打 log
            if update_count % args.log_interval == 0:
                if args.save:
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename)
                print(
                    f'Update #{update_count:05d} | Recon Error: {recon_loss.item():.2f} '
                    f'| KL: {kl_loss.item():.2f} | Total Loss: {loss.item():.2f}'
                )

            update_count += 1

            # 定期做验证
            if update_count % (args.log_interval * 5) == 0:
                evaluate(model, validation_loader)

            if update_count >= args.n_updates:
                break

    pbar.close()


if __name__ == "__main__":
    train()
