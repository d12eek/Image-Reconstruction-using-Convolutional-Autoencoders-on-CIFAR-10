# ============================================================
# evaluate.py
# ============================================================

print("Step 1: Starting...")

import os
import sys

print("Step 2: os, sys imported...")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(SRC_DIR)

print(f"Step 3: Paths set... ROOT={ROOT_DIR}")

import torch
print("Step 4: torch imported...")

import numpy as np
print("Step 5: numpy imported...")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
print("Step 6: matplotlib imported...")

from pytorch_msssim import ssim
print("Step 7: ssim imported...")

from model import ConvAutoencoder
print("Step 8: model imported...")

from dataset import get_debug_dataloader
print("Step 9: dataset imported...")

from train import train_model
print("Step 10: train imported...")


def evaluate_model(model, images, labels_list, class_names, device):

    print("\n🔍 Running evaluation...")
    model.eval()

    with torch.no_grad():
        images_device = images.to(device)
        recon = model(images_device)

    images_cpu = images.cpu()
    recon_cpu  = recon.cpu()

    print("\n📊 Similarity Scores Per Class:")
    print("=" * 55)
    print(f"  {'Class':<12} {'MSE':>10} {'SSIM':>10}")
    print("=" * 55)

    mse_scores  = []
    ssim_scores = []

    for i in range(len(images_cpu)):
        mse = torch.mean((images_cpu[i] - recon_cpu[i]) ** 2).item()

        img_batch  = images_cpu[i].unsqueeze(0)
        rec_batch  = recon_cpu[i].unsqueeze(0)
        ssim_score = ssim(
            img_batch, rec_batch,
            data_range=1.0,
            size_average=True
        ).item()

        mse_scores.append(mse)
        ssim_scores.append(ssim_score)

        print(f"  {class_names[labels_list[i]]:<12} "
              f"{mse:>10.5f} {ssim_score:>10.4f}")

    print("=" * 55)
    print(f"  {'AVERAGE':<12} "
          f"{np.mean(mse_scores):>10.5f} "
          f"{np.mean(ssim_scores):>10.4f}")
    print("=" * 55)

    # ── Save scores ───────────────────────────────────────────
    scores_path = os.path.join(OUTPUT_DIR, 'similarity_scores.txt')
    with open(scores_path, 'w') as f:
        f.write(f"{'Class':<12} {'MSE':>10} {'SSIM':>10}\n")
        for i in range(len(images_cpu)):
            f.write(f"{class_names[labels_list[i]]:<12} "
                    f"{mse_scores[i]:>10.5f} "
                    f"{ssim_scores[i]:>10.4f}\n")
        f.write(f"{'AVERAGE':<12} "
                f"{np.mean(mse_scores):>10.5f} "
                f"{np.mean(ssim_scores):>10.4f}\n")
    print(f"\n💾 Scores saved → {scores_path}")

    # ── Visualization ─────────────────────────────────────────
    print("🎨 Creating visualization...")
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle(
        'Convolutional Autoencoder — Original vs Reconstructed',
        fontsize=13, fontweight='bold'
    )

    for i in range(10):
        orig = images_cpu[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig)
        axes[0, i].set_title(class_names[labels_list[i]], fontsize=9)
        axes[0, i].axis('off')

        rec = np.clip(recon_cpu[i].permute(1, 2, 0).numpy(), 0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].set_title(f'SSIM:{ssim_scores[i]:.3f}', fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Reconstructed', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(OUTPUT_DIR, 'reconstruction.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"🖼️  Plot saved → {save_path}")
    print("\n✅ Evaluation Complete!")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Training model for 50 epochs...")
    model, images, labels, class_names, device, losses = train_model(epochs=50)
    labels_list = labels.tolist()
    evaluate_model(model, images, labels_list, class_names, device)