import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import CellImageHeatmapDataset
from model import SimpleCNN

# === Parameter ===
DATASET_PATH = 'data/dataset'
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Daten laden ===
dataset = CellImageHeatmapDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Modell ===
model = SimpleCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training ===
losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, heatmaps in dataloader:
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.6f}")

    # === Visualisierung alle 5 Epochen ===
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample_img, sample_heat = dataset[0]
            pred = model(sample_img.unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(sample_img.squeeze(), cmap='gray')
            plt.title('Input Image')
            plt.subplot(1, 3, 2)
            plt.imshow(sample_heat.squeeze(), cmap='hot')
            plt.title('Target Heatmap')
            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='hot')
            plt.title('Prediction')
            plt.tight_layout()
            plt.show()

# === Verlustverlauf speichern/plotten ===
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid()
plt.show()

# === Modell speichern (optional) ===
torch.save(model.state_dict(), 'cell_model.pth')
print("✅ Modell gespeichert als 'cell_model.pth'")
