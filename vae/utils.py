import os
import time
import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_dir = pt_dir
        self.file_names = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.pt_dir, self.file_names[idx])
        tensor = torch.load(file_path)  # 应该是 [1, 3, 200, 200]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor.squeeze(0)  # 变成 [3, 200, 200]


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
