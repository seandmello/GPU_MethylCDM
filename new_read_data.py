import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np


class PatchRNADataset(Dataset):
    def __init__(self, patch_root, rna_root=None, cancer_types=None,
                 max_patches_per_wsi=None, transform=None):
        self.samples = []
        self.has_rna = rna_root is not None
        self.has_cancer_type = cancer_types is not None
        self.num_cancer_types = len(cancer_types) if cancer_types else 0

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        if self.has_cancer_type:
            # Multi-folder cancer-type conditioned mode
            # cancer_types is a list like ["TCGA-BLCA", "TCGA-BRCA", ...]
            # patch_root is the parent dir containing these folders
            for ct_idx, ct_name in enumerate(cancer_types):
                ct_dir = os.path.join(patch_root, ct_name)
                if not os.path.isdir(ct_dir):
                    print(f"Warning: cancer type folder not found: {ct_dir}")
                    continue

                for wsi in os.listdir(ct_dir):
                    wsi_dir = os.path.join(ct_dir, wsi)
                    if not os.path.isdir(wsi_dir):
                        continue

                    patches = [os.path.join(wsi_dir, p) for p in os.listdir(wsi_dir)]
                    if max_patches_per_wsi is not None and len(patches) > max_patches_per_wsi:
                        patches = random.sample(patches, max_patches_per_wsi)

                    for patch_path in patches:
                        self.samples.append((patch_path, ct_idx))

        else:
            # Original single-folder mode (unconditional or RNA-conditioned)
            for wsi in os.listdir(patch_root):
                patch_dir = os.path.join(patch_root, wsi)

                if not os.path.isdir(patch_dir):
                    continue

                rna_path = None
                if self.has_rna:
                    rna_path = os.path.join(rna_root, f"{wsi}.npy")
                    if not os.path.exists(rna_path):
                        continue

                patches = [os.path.join(patch_dir, p) for p in os.listdir(patch_dir)]
                if max_patches_per_wsi is not None and len(patches) > max_patches_per_wsi:
                    patches = random.sample(patches, max_patches_per_wsi)

                for patch_path in patches:
                    self.samples.append((patch_path, rna_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_path, cond = self.samples[idx]

        try:
            image = Image.open(patch_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Corrupted file — return a random valid sample instead
            return self[random.randint(0, len(self) - 1)]

        if self.transform:
            image = self.transform(image)

        out = {"image": image}
        if self.has_cancer_type:
            one_hot = torch.zeros(self.num_cancer_types)
            one_hot[cond] = 1.0
            out["methyl_data"] = one_hot
        elif self.has_rna:
            out["methyl_data"] = torch.from_numpy(np.load(cond)).float()
        return out
