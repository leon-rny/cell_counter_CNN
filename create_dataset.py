import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

def augment_and_save_dataset(patch_dir, output_dir, num_augs=4):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy') and 'heatmap' not in f])
    count = 0

    for f in files:
        base = f[:-4]
        patch_path = os.path.join(patch_dir, f)
        heatmap_path = os.path.join(patch_dir, base + '_heatmap.npy')

        if not os.path.exists(heatmap_path):
            continue

        img = np.load(patch_path).astype(np.float32)
        heat = np.load(heatmap_path).astype(np.float32)

        # [1, H, W] für torchvision
        img_t = torch.from_numpy(img).unsqueeze(0)
        heat_t = torch.from_numpy(heat).unsqueeze(0)

        for _ in range(num_augs):
            a_img, a_heat = img_t.clone(), heat_t.clone()

            if random.random() < 0.5:
                a_img = TF.hflip(a_img)
                a_heat = TF.hflip(a_heat)
            if random.random() < 0.5:
                a_img = TF.vflip(a_img)
                a_heat = TF.vflip(a_heat)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                a_img = TF.rotate(a_img, angle)
                a_heat = TF.rotate(a_heat, angle)

            # zurück in numpy
            np.save(os.path.join(output_dir, f'sample_{count:04d}_img.npy'), a_img.squeeze(0).numpy())
            np.save(os.path.join(output_dir, f'sample_{count:04d}_heat.npy'), a_heat.squeeze(0).numpy())
            count += 1

def augment_all_patch_folders(parent_patch_dir, output_dir, num_augs=4):
    os.makedirs(output_dir, exist_ok=True)
    total = 0

    subfolders = sorted([f for f in os.listdir(parent_patch_dir) if os.path.isdir(os.path.join(parent_patch_dir, f))])

    for folder in subfolders:
        full_path = os.path.join(parent_patch_dir, folder)
        print(f"🔄 Bearbeite: {folder}")
        augment_and_save_dataset(
            patch_dir=full_path,
            output_dir=output_dir,
            num_augs=num_augs
        )
        total += 1

cwd = os.getcwd()
patches_path = cwd + '/data/patches'
output_dir = cwd + '/data/dataset'

augment_all_patch_folders(parent_patch_dir= patches_path, output_dir=output_dir, num_augs=4)
