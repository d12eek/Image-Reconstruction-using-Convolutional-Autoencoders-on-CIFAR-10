# ============================================================
# dataset.py — Load CIFAR-10 & Pick 1 Image Per Class
# ============================================================

import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ── Always point to root/data/ folder ────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA_DIR = os.path.normpath(DATA_DIR)

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_debug_dataloader(batch_size=10):
    """
    Loads CIFAR-10 from root/data/ folder.
    Selects exactly 1 image per class = 10 images total.
    Downloads only if not already present.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ── Download only if not already present ─────────────────
    full_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,      # Only downloads if not present
        transform=transform
    )

    # ── Pick exactly 1 image per class ───────────────────────
    selected_indices = []
    found_classes = {}

    for idx, (_, label) in enumerate(full_dataset):
        if label not in found_classes:
            found_classes[label] = idx
            selected_indices.append(idx)
        if len(found_classes) == 10:
            break

    # Sort by class label 0→9
    selected_indices.sort(key=lambda i: full_dataset[i][1])

    # Create subset with only 10 images
    debug_dataset = Subset(full_dataset, selected_indices)

    # ── Print selected images ─────────────────────────────────
    print("\n📦 Debug Dataset — 1 Image Per Class:")
    print("-" * 35)
    for idx in selected_indices:
        img, label = full_dataset[idx]
        print(f"  Class {label:2d} → {CLASS_NAMES[label]:12s} | Index: {idx}")
    print("-" * 35)
    print(f"  Total images selected: {len(debug_dataset)}\n")

    dataloader = DataLoader(
        debug_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader, CLASS_NAMES