import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import CellImageHeatmapDataset
from model import MiniUNet

# Parameter
VAL_DIR = 'data/dataset/val'           
MODEL_PATH = 'cell_miniunet.pth'   
BATCH_SIZE = 1                          
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load dataset
val_dataset = CellImageHeatmapDataset(VAL_DIR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# evalation
n_samples_to_show = 5

with torch.no_grad():
    for idx, (imgs, heatmaps) in enumerate(val_loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)

        # tensor to numpy for plotting
        img_np = imgs.squeeze().cpu().numpy()
        heatmap_np = heatmaps.squeeze().cpu().numpy()
        pred_np = preds.squeeze().cpu().numpy()

        # Plots
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title('Input Patch')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='hot')
        plt.title('Ground Truth Heatmap')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_np, cmap='hot')
        plt.title('Predicted Heatmap')
        plt.axis('off')

        plt.suptitle(f'Sample {idx+1}')
        plt.tight_layout()
        plt.show()

        if idx + 1 >= n_samples_to_show:
            break