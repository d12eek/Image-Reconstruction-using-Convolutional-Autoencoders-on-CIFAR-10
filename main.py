# ============================================================
# main.py — Runs Everything End to End
# ============================================================

print("Starting...")

import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

print("Importing modules...")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

print("All modules imported!")

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Class Names ───────────────────────────────────────────────
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ============================================================
# SECTION 1 — DATASET
# ============================================================
print("\n" + "=" * 50)
print("         SECTION 1: DATASET")
print("=" * 50)

transform = transforms.Compose([transforms.ToTensor()])

full_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

# Pick 1 image per class
selected_indices = []
found_classes    = {}

for idx, (_, label) in enumerate(full_dataset):
    if label not in found_classes:
        found_classes[label] = idx
        selected_indices.append(idx)
    if len(found_classes) == 10:
        break

selected_indices.sort(key=lambda i: full_dataset[i][1])
debug_dataset = Subset(full_dataset, selected_indices)

print("\n📦 Debug Dataset — 1 Image Per Class:")
print("-" * 35)
for idx in selected_indices:
    _, label = full_dataset[idx]
    print(f"  Class {label} → {CLASS_NAMES[label]:12s} | Index: {idx}")
print("-" * 35)
print(f"  Total: {len(debug_dataset)} images\n")

dataloader = DataLoader(debug_dataset, batch_size=10, shuffle=False)

# ============================================================
# SECTION 2 — MODEL
# ============================================================
print("=" * 50)
print("         SECTION 2: MODEL")
print("=" * 50)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = ConvAutoencoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🖥️  Device      : {device}")
print(f"🧠 Parameters  : {total_params:,}")

# ============================================================
# SECTION 3 — TRAINING
# ============================================================
print("\n" + "=" * 50)
print("         SECTION 3: TRAINING")
print("=" * 50)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs    = 200

print(f"\n📉 Loss     : MSELoss")
print(f"⚙️  Optimizer: Adam (lr=0.001)")
print(f"🔁 Epochs   : {epochs}")
print(f"🖼️  Images   : 10 (1 per class)\n")

model.train()
for epoch in range(1, epochs + 1):
    for images, _ in dataloader:
        images        = images.to(device)
        reconstructed = model(images)
        loss          = criterion(reconstructed, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:2d}/{epochs}]  →  Loss: {loss.item():.6f}")

print("\n✅ Training Complete!")

# ============================================================
# SECTION 4 — EVALUATION
# ============================================================
print("\n" + "=" * 50)
print("         SECTION 4: EVALUATION")
print("=" * 50)

model.eval()
images_batch, labels_batch = next(iter(dataloader))

with torch.no_grad():
    images_device = images_batch.to(device)
    recon         = model(images_device)

images_cpu = images_batch.cpu()
recon_cpu  = recon.cpu()
labels_list = labels_batch.tolist()

print("\n📊 Similarity Scores:")
print("=" * 55)
print(f"  {'Class':<12} {'MSE':>10} {'SSIM':>10} {'Quality':>12}")
print("=" * 55)

mse_scores  = []
ssim_scores = []

for i in range(10):
    mse = torch.mean((images_cpu[i] - recon_cpu[i]) ** 2).item()
    ssim_score = ssim(
        images_cpu[i].unsqueeze(0),
        recon_cpu[i].unsqueeze(0),
        data_range=1.0,
        size_average=True
    ).item()
    mse_scores.append(mse)
    ssim_scores.append(ssim_score)

    quality = "🟢 Excellent" if ssim_score > 0.85 else "🟡 Good" if ssim_score > 0.70 else "🔴 Poor"
    print(f"  {CLASS_NAMES[labels_list[i]]:<12} {mse:>10.5f} {ssim_score:>10.4f} {quality}")

print("=" * 55)
print(f"  {'AVERAGE':<12} {np.mean(mse_scores):>10.5f} {np.mean(ssim_scores):>10.4f}")
print("=" * 55)

# Save scores
scores_path = os.path.join(OUTPUT_DIR, 'similarity_scores.txt')
with open(scores_path, 'w') as f:
    f.write(f"{'Class':<12} {'MSE':>10} {'SSIM':>10}\n")
    f.write("=" * 35 + "\n")
    for i in range(10):
        f.write(f"{CLASS_NAMES[labels_list[i]]:<12} {mse_scores[i]:>10.5f} {ssim_scores[i]:>10.4f}\n")
    f.write("=" * 35 + "\n")
    f.write(f"{'AVERAGE':<12} {np.mean(mse_scores):>10.5f} {np.mean(ssim_scores):>10.4f}\n")
print(f"\n💾 Scores saved → {scores_path}")

# ============================================================
# SECTION 5 — VISUALIZATION
# ============================================================
print("\n" + "=" * 50)
print("         SECTION 5: VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(2, 10, figsize=(20, 5))
fig.suptitle(
    'Convolutional Autoencoder — Original vs Reconstructed\nCIFAR-10 (1 Image Per Class)',
    fontsize=13, fontweight='bold'
)

for i in range(10):
    orig = images_cpu[i].permute(1, 2, 0).numpy()
    axes[0, i].imshow(orig)
    axes[0, i].set_title(CLASS_NAMES[labels_list[i]], fontsize=9, fontweight='bold')
    axes[0, i].axis('off')

    rec = np.clip(recon_cpu[i].permute(1, 2, 0).numpy(), 0, 1)
    axes[1, i].imshow(rec)
    axes[1, i].set_title(f'SSIM:{ssim_scores[i]:.3f}', fontsize=8, color='green')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Reconstructed', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'reconstruction.png')
plt.savefig(save_path, bbox_inches='tight', dpi=150)
print(f"\n🖼️  Plot saved → {save_path}")
print("\n✅ All Done! Check output/ folder for results!")