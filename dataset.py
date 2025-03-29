import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd

class CellDataset(Dataset):
    def __init__(self, image, csv_path, sigma=2):
        self.image = image.astype(np.float32)[None, ...]
        df = pd.read_csv(csv_path)
        self.points = df[['x', 'y']].values
        self.heatmap = self.generate_heatmap(self.points, image.shape, sigma)

    def generate_heatmap(self, points, shape, sigma):
        heatmap = np.zeros(shape, dtype=np.float32)
        for x, y in points:
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                heatmap[int(y), int(x)] = 1.0
        return gaussian_filter(heatmap, sigma=sigma)[None, ...]

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        return torch.tensor(self.image), torch.tensor(self.heatmap)
