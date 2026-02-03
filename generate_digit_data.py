import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

# --- Your Dataset Class ---
class PatchRNADataset(Dataset):
    def __init__(self, patch_root, rna_root, transform=None):
        self.samples = []
        self.transform = transform

        for wsi in os.listdir(patch_root):
            patch_dir = os.path.join(patch_root, wsi)
            rna_path = os.path.join(rna_root, f"{wsi}.npy")

            if not os.path.isdir(patch_dir):
                continue
            if not os.path.exists(rna_path):
                continue

            for patch_name in os.listdir(patch_dir):
                patch_path = os.path.join(patch_dir, patch_name)
                self.samples.append((patch_path, rna_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_path, rna_path = self.samples[idx]

        image = Image.open(patch_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        rna = torch.from_numpy(np.load(rna_path)).float()

        return {
            "image": image,
            "rna_data": rna
        }

# --- Prepare MNIST Patches and Random RNA ---
data_root = "/cluster/projects/kumargroup/sean/Methylation_Generation/MNIST"
patch_root = os.path.join(data_root, "mnist_patches")
rna_root = os.path.join(data_root, "rna_data")

os.makedirs(patch_root, exist_ok=True)
os.makedirs(rna_root, exist_ok=True)

# Load MNIST from pre-downloaded cache (no internet required)
# torchvision expects {root}/MNIST/raw/, so set root one level above
mnist = datasets.MNIST(root="/cluster/projects/kumargroup/sean/Methylation_Generation", train=True, download=False)

# Group all MNIST indices by their digit label
digit_indices = {d: [] for d in range(10)}
for idx in range(len(mnist)):
    _, label = mnist[idx]
    digit_indices[label].append(idx)

num_images_per_folder = 1300

# One folder per digit (0-9), each folder contains only that digit
for digit in range(10):
    wsi_name = f"WSI_{digit}"
    wsi_dir = os.path.join(patch_root, wsi_name)
    os.makedirs(wsi_dir, exist_ok=True)

    # Pick random images of this digit
    chosen = random.sample(digit_indices[digit], num_images_per_folder)
    for i, idx in enumerate(chosen):
        img, label = mnist[idx]
        img_path = os.path.join(wsi_dir, f"patch_{i}.png")
        img.save(img_path)

    # Methyl embedding is just the digit value repeated 10 times
    rna_vector = np.full(10, float(digit))
    np.save(os.path.join(rna_root, f"{wsi_name}.npy"), rna_vector)

# --- Create Dataset and DataLoader ---
dataset = PatchRNADataset(
    patch_root=patch_root,
    rna_root=rna_root,
    transform=transforms.ToTensor()
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Iterate over batches ---
for batch in loader:
    print("Image batch shape:", batch["image"].shape)
    print("RNA batch shape:", batch["rna_data"].shape)
    break
