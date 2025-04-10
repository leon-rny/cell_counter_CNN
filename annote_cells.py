import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import czifile
from roifile import ImagejRoi


def load_image(image_path, roi_path):
    with czifile.CziFile(image_path) as czi:
        image = czi.asarray()
    image = np.squeeze(image)

    roi = ImagejRoi.fromfile(roi_path)
    roi_coords = roi.coordinates()
    closed_coords = np.vstack([roi_coords, roi_coords[0]])

    # Normalize the second channel
    channel_2 = image[1] * 25
    normalized = (channel_2 - np.min(channel_2)) / (np.max(channel_2) - np.min(channel_2))
    return normalized, closed_coords

def annotate_points(image, coords, output_csv):
    points = []

    def onclick(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, 'ro')
            fig.canvas.draw()

    # Display the image and ROI
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='Reds')
    ax.plot(coords[:, 0], coords[:, 1], 'r--')

    points = []
    ax.set_title("Click on the image to select points")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    df = pd.DataFrame(points, columns=['x', 'y'])
    df.to_csv(output_csv, index=False)
    print(f"{len(points)} points saved as: {output_csv}")

def create_patches(image, points_csv, patch_size=100, output_dir='patches', number='01_3'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(points_csv)
    points = df[['x', 'y']].values.astype(int)

    h, w = image.shape
    half = patch_size // 2
    patch_id = 0

    for x, y in points:
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half

        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue

        patch = image[y0:y1, x0:x1]
        local_point = [(half, half)]

        patch_path = os.path.join(output_dir, f"{number}_{patch_id:04d}.npy")
        coords_path = os.path.join(output_dir, f"{number}_{patch_id:04d}_coords.csv")

        np.save(patch_path, patch)
        pd.DataFrame(local_point, columns=['x', 'y']).to_csv(coords_path, index=False)

        patch_id += 1

def generate_heatmap(shape, points, sigma=2):
    heatmap = np.zeros(shape, dtype=np.float32)
    for x, y in points:
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            heatmap[int(y), int(x)] = 1.0
    return gaussian_filter(heatmap, sigma=sigma)

def create_heatmaps_for_patches(patch_dir, sigma=2):
    count = 0
    for file in sorted(os.listdir(patch_dir)):
        if not file.endswith('.npy') or 'heatmap' in file:
            continue

        base = file[:-4]
        patch_path = os.path.join(patch_dir, file)
        coords_path = os.path.join(patch_dir, base + '_coords.csv')
        heatmap_path = os.path.join(patch_dir, base + '_heatmap.npy')

        patch = np.load(patch_path)
        df = pd.read_csv(coords_path)
        points = df[['x', 'y']].values

        heatmap = generate_heatmap(patch.shape, points, sigma)
        np.save(heatmap_path, heatmap)
        count += 1

# Paths
cwd = os.getcwd()
image_path = cwd + '/data/images'
roi_path = cwd + '/data/rois'
patches_path = cwd + '/data/patches'
output_csv_path = cwd + '/data/coords'

# filter for .czi files
image_files = [f for f in os.listdir(image_path) if f.endswith('.czi')]
roi_files = [f for f in os.listdir(roi_path) if f.endswith('.roi')]

for i in range(1, 6):
    image_file = image_files[i]
    roi_file = roi_files[i]
    image_path = os.path.join(image_path, image_file)
    roi_path = os.path.join(roi_path, roi_file)
    number = image_file[-8: -4]
    output_csv_path = os.path.join(output_csv_path, f"{number}_coords.csv")
    patches_path = os.path.join(patches_path, number)

    # Load image and ROI
    normalized, closed_coords = load_image(image_path, roi_path)

    # Annotate points
    annotate_points(normalized, closed_coords, output_csv_path)

    # Create patches
    create_patches(image=normalized, points_csv=output_csv_path, patch_size=100, output_dir=patches_path, number=number)

    # reset paths
    image_path = cwd + '/data/images'
    roi_path = cwd + '/data/rois'
    patches_path = cwd + '/data/patches'
    output_csv_path = cwd + '/data/coords'
