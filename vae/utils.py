import os
import time
import torch
from torch.utils.data import Dataset

import os
import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, pt_dir):
        self.files = sorted(
            [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith('.pt')]
        )
        # —— ① 预扫一遍文件，统计全局 min/max ——
        all_mins, all_maxs = [], []
        for f in self.files:
            t = torch.load(f, map_location="cpu", weights_only=True).float()
            all_mins.append(t.min())
            all_maxs.append(t.max())
        self.x_min = torch.stack(all_mins).min()
        self.x_max = torch.stack(all_maxs).max()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [200, 200] → [1, 200, 200]
        elif x.shape[0] != 1:
            raise ValueError(f"Unexpected channel size: {x.shape}")
        x = (x - self.x_min) / (self.x_max - self.x_min)
        return x.float()


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/model' + timestamp + '.pth')
