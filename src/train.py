# ============================================================
# train.py — Training Loop for Convolutional Autoencoder
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# ── Add src/ to path so imports work ─────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from model import ConvAutoencoder
from dataset import get_debug_dataloader


def train_model(epochs=10, learning_rate=1e-3):

    # ── Device Setup ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device       : {device}")

    # ── Load Dataset ─────────────────────────────────────────
    dataloader, class_names = get_debug_dataloader(batch_size=10)

    # ── Model, Loss, Optimizer ───────────────────────────────
    model     = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"📉 Loss Function : MSELoss")
    print(f"⚙️  Optimizer     : Adam (lr={learning_rate})")
    print(f"🔁 Epochs        : {epochs}")
    print(f"🖼️  Images        : 10 (1 per class)\n")
    print("=" * 45)
    print("         TRAINING STARTED")
    print("=" * 45)

    # ── Training Loop ────────────────────────────────────────
    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)

            # Forward Pass
            reconstructed = model(images)

            # Compute Loss
            loss = criterion(reconstructed, images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch [{epoch:2d}/{epochs}]  →  Loss: {avg_loss:.6f}")

    print("=" * 45)
    print("✅ Training Complete!\n")

    # ── Return for evaluation ─────────────────────────────────
    final_images, final_labels = next(iter(dataloader))
    return model, final_images, final_labels, class_names, device, epoch_losses


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model, images, labels, class_names, device, losses = train_model(epochs=10)
    print(f"📦 Image batch shape : {images.shape}")
    print(f"📦 Labels            : {labels.tolist()}")
    print(f"📉 Final loss        : {losses[-1]:.6f}")