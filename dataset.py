import torch
from torch.utils.data import Dataset
import os
import numpy as np

class CellImageHeatmapDataset(Dataset):
    def __init__(self, data_dir):
        """
        Expects folder-structur:
        - data_dir/
            - sample_XXXX_img.npy
            - sample_XXXX_heat.npy
        """
        self.data_dir = data_dir
        self.sample_ids = sorted(set(f.split('_')[1] for f in os.listdir(data_dir) if f.endswith('_img.npy')))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        img_path = os.path.join(self.data_dir, f'sample_{sid}_img.npy')
        heat_path = os.path.join(self.data_dir, f'sample_{sid}_heat.npy')

        img = np.load(img_path).astype(np.float32)  # [H, W]
        heat = np.load(heat_path).astype(np.float32)  # [H, W]

        # in tensor [1, H, W]
        img = torch.from_numpy(img).unsqueeze(0)
        heat = torch.from_numpy(heat).unsqueeze(0)

        return img, heat
