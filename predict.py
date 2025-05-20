import numpy as np
import torch
from czifile import CziFile
from roifile import ImagejRoi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import peak_local_max
from skimage.draw import polygon
from skimage.measure import label, regionprops

from model import MiniUNet

def load_image(image_path, roi_path):
    # load image
    with CziFile(image_path) as czi:
        img = czi.asarray()

    img = np.squeeze(img)
    channel_2 = img[1]
    channel_2 = (channel_2 - np.min(channel_2)) / (np.max(channel_2) - np.min(channel_2))
    image = channel_2.astype(np.float32)
    
    # load roi
    roi = ImagejRoi.fromfile(roi_path)
    roi_coords = roi.coordinates()
    closed_coords = np.vstack([roi_coords, roi_coords[0]])
    mask = np.zeros_like(image, dtype=bool)
    rr, cc = polygon(closed_coords[:, 1], closed_coords[:, 0], shape=mask.shape)
    mask[rr, cc] = True

    masked_image = mask * image
    return masked_image

# parameter
tissue_type = 'duodenum_06_7'
MODEL_PATH = 'model/cell_miniunet.pth'                           
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CZI_PATH = rf'data/images/{tissue_type}.czi'
ROI_PATH = rf'data/rois/{tissue_type}.roi'
PATCH_SIZE = 100
STRIDE = 50

# load image
image = load_image(CZI_PATH, ROI_PATH)
H, W = image.shape

# load model
model = MiniUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# prepare heatmaps
full_heatmap = np.zeros_like(image)
weight_map = np.zeros_like(image)
patch_coords = []

# sliding-window prediction
for y0 in range(0, H - PATCH_SIZE + 1, STRIDE):
    for x0 in range(0, W - PATCH_SIZE + 1, STRIDE):
        patch = image[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(patch_tensor).squeeze().cpu().numpy()

        full_heatmap[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += pred
        weight_map[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0

normalized_heatmap = full_heatmap / np.maximum(weight_map, 1e-6)

coords = peak_local_max(normalized_heatmap, threshold_abs=0.002, min_distance=5)
print(f"Found cells: {len(coords)}")

# plot
threshold = 0.002
binary_map = normalized_heatmap > threshold

label_map = label(binary_map)
regions = regionprops(label_map)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(image, cmap='Reds')
ax.set_title('Celldetection')
ax.axis('off')

for region in regions:
    if region.area < 10:
        continue

    # Bounding Box
    minr, minc, maxr, maxc = region.bbox
    # Increase rectangle size by expanding 5 pixels in each direction
    expand = 5
    minr_exp = max(minr - expand, 0)
    minc_exp = max(minc - expand, 0)
    maxr_exp = min(maxr + expand, image.shape[0])
    maxc_exp = min(maxc + expand, image.shape[1])

    rect = patches.Rectangle(
        (minc_exp, minr_exp),
        maxc_exp - minc_exp,
        maxr_exp - minr_exp,
        linewidth=1.5,
        edgecolor='lime',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig('output.png', dpi=500)
plt.show()