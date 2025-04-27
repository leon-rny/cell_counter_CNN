import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import train_test_split

def load_and_preprocess_patch(patch_path, heatmap_path, heatmap_factor=5.0):
    img = np.load(patch_path).astype(np.float32)
    heat = np.load(heatmap_path).astype(np.float32)

    # Normalize image 0-1 (min-max)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Strengthen heatmap
    heat *= heatmap_factor
    heat = np.clip(heat, 0, 1)

    return img, heat

def augment_patch(img, heat):
    img_t = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
    heat_t = torch.from_numpy(heat).unsqueeze(0)

    if random.random() < 0.5:
        img_t = TF.hflip(img_t)
        heat_t = TF.hflip(heat_t)
    if random.random() < 0.5:
        img_t = TF.vflip(img_t)
        heat_t = TF.vflip(heat_t)
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_t = TF.rotate(img_t, angle)
        heat_t = TF.rotate(heat_t, angle)

    return img_t.squeeze(0).numpy(), heat_t.squeeze(0).numpy()

def create_dataset(parent_patch_dir, output_base_dir, num_augs=4, val_split=0.2):  
    os.makedirs(output_base_dir, exist_ok=True)
    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_patches = []

    # Collect all base names
    subfolders = sorted([f for f in os.listdir(parent_patch_dir) if os.path.isdir(os.path.join(parent_patch_dir, f))])
    for folder in subfolders:
        folder_path = os.path.join(parent_patch_dir, folder)
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy') and 'heatmap' not in f])

        for f in files:
            base = f[:-4]
            patch_path = os.path.join(folder_path, f)
            heatmap_path = os.path.join(folder_path, base + '_heatmap.npy')

            if os.path.exists(patch_path) and os.path.exists(heatmap_path):
                all_patches.append((patch_path, heatmap_path))

    print(f"Found Patches: {len(all_patches)}")

    # Train/Validation Split
    train_patches, val_patches = train_test_split(all_patches, test_size=val_split, random_state=42)

    def save_augmented(patches, out_dir, start_idx=0):
        count = start_idx
        for patch_path, heatmap_path in patches:
            img, heat = load_and_preprocess_patch(patch_path, heatmap_path)

            for _ in range(num_augs):
                a_img, a_heat = augment_patch(img, heat)

                np.save(os.path.join(out_dir, f'sample_{count:04d}_img.npy'), a_img)
                np.save(os.path.join(out_dir, f'sample_{count:04d}_heat.npy'), a_heat)
                count += 1
        return count

    # Save train and val sets
    c1 = save_augmented(train_patches, train_dir)
    c2 = save_augmented(val_patches, val_dir, start_idx=c1)

    print(f"Train Samples: {c1}, Validation Samples: {c2 - c1}")

cwd = os.getcwd()
patches_path = cwd + '/data/patches'
output_dir = cwd + '/data/dataset'

create_dataset(parent_patch_dir=patches_path, output_base_dir=output_dir, num_augs=4)
