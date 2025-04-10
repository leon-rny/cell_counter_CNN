import numpy as np
import torch
import matplotlib.pyplot as plt
import czifile
from skimage.feature import peak_local_max
from model import SimpleCNN
from scipy.ndimage import gaussian_filter

# === PARAMETER ===
PATCH_SIZE = 100
STRIDE = 50
SIGMA = 2
THRESHOLD = 0.3
MODEL_PATH = 'cell_model.pth'
CZI_PATH = 'data/images/duodenum_01_3.czi'  # <- anpassen

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === MODELL LADEN ===
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === CZI BILD LADEN UND NORMALISIEREN ===
with czifile.CziFile(CZI_PATH) as czi:
    img = czi.asarray()
img = np.squeeze(img)
channel_2 = img[1]  # Kanal 2 = Index 1
channel_2 = (channel_2 - channel_2.min()) / (channel_2.max() - channel_2.min())
image = channel_2.astype(np.float32)

H, W = image.shape
heatmap_full = np.zeros((H, W), dtype=np.float32)

# === PATCH-WEISE PREDICTION ===
for y0 in range(0, H - PATCH_SIZE + 1, STRIDE):
    for x0 in range(0, W - PATCH_SIZE + 1, STRIDE):
        patch = image[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(patch_tensor).squeeze().cpu().numpy()

        # Glätten optional
        pred = gaussian_filter(pred, sigma=1)

        # In das Gesamtbild einsetzen (mit MaxPooling-Effekt)
        heatmap_full[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = np.maximum(
            heatmap_full[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE],
            pred
        )

# === PEAK DETECTION ===
coords = peak_local_max(heatmap_full, threshold_abs=THRESHOLD, min_distance=5)
print(f"✅ {len(coords)} Zellen erkannt")

# === VISUALISIERUNG ===
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.scatter(coords[:, 1], coords[:, 0], c='red', s=10)
plt.title(f"{len(coords)} erkannte Zellen")
plt.axis('off')
plt.tight_layout()
plt.show()
