import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import SimpleCNN
from dataset import CellDataset
import matplotlib.pyplot as plt
import czifile

cwd = os.getcwd()
number = '01_3'
image_path = cwd + f'/data/duodenum_{number}.czi'

with czifile.CziFile(image_path) as czi:
    image = czi.asarray()
image = np.squeeze(image)

channel_2 = image[1] * 25
normalized = (channel_2 - np.min(channel_2)) / (np.max(channel_2) - np.min(channel_2))

dataset = CellDataset(image=normalized, csv_path='data/cell_points_01_3.csv')
loader = DataLoader(dataset, batch_size=1)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(50):
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

with torch.no_grad():
    pred = model(x).squeeze().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(normalized, cmap='gray')
    ax[1].imshow(pred, cmap='viridis')
    plt.show()
