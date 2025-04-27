import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CellImageHeatmapDataset
from model import MiniUNet

# Parameter
TRAIN_DIR = 'data/dataset/train'
VAL_DIR = 'data/dataset/val'
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load dataset
train_dataset = CellImageHeatmapDataset(TRAIN_DIR)
val_dataset = CellImageHeatmapDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}")

# create model
model = MiniUNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# training and validation loop
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # train
    for imgs, heatmaps in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

    # validate
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for imgs, heatmaps in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, heatmaps)

            val_running_loss += loss.item() * imgs.size(0)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    # save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'model/cell_miniunet.pth')
        print(f"Saved best model (Val Loss: {best_val_loss:.6f})")


# Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.title('Training vs Validation Loss')
plt.show()
