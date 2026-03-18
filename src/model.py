# ============================================================
# model.py — Convolutional Autoencoder Architecture
# ============================================================

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input: (3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # → (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → (128, 4, 4)
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),        # → (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),        # → (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),        # → (3, 32, 32)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded