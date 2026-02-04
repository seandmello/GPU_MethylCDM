import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np

class PatchRNADataset(Dataset):
    def __init__(self, patch_root, rna_root=None, transform=None):
        self.samples = []
        self.has_rna = rna_root is not None
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        for wsi in os.listdir(patch_root):
            patch_dir = os.path.join(patch_root, wsi)

            if not os.path.isdir(patch_dir):
                continue

            rna_path = None
            if self.has_rna:
                rna_path = os.path.join(rna_root, f"{wsi}.npy")
                if not os.path.exists(rna_path):
                    continue

            for patch_name in os.listdir(patch_dir):
                patch_path = os.path.join(patch_dir, patch_name)
                self.samples.append((patch_path, rna_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_path, rna_path = self.samples[idx]

        try:
            image = Image.open(patch_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Corrupted file — return a random valid sample instead
            return self[random.randint(0, len(self) - 1)]

        if self.transform:
            image = self.transform(image)

        out = {"image": image}
        if self.has_rna:
            out["methyl_data"] = torch.from_numpy(np.load(rna_path)).float()
        return out
