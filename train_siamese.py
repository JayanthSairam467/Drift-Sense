"""
Drift-Sense: Siamese Network Training Script (v2.1 — Bug Fixes)
================================================================
Trains the SiameseRanker on synthetic data generated ON-THE-FLY.

CRITICAL FIXES from v2.0:
  1. sample.reference_img → sample['reference_img'] (dict access bug)
  2. Fixed PIL image handling (keep uint8 for resize, normalize after)
  3. Negatives now generated at PERIODIC OFFSETS (real ambiguity source)
  4. Proper np.random.Generator usage matching pipeline API
  5. More epochs and samples for better convergence

Architecture:
  - Reference: 1000x1000 uint8 → downsample to 100x100
  - Positive: 100x100 crop from search at true location
  - Negative: 100x100 crop from search at WRONG periodic offset
  - Loss: TripletMarginLoss (embedding distance based)
"""

import argparse
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from siamese_net import SiameseRanker, export_to_onnx
from src.pipeline import GenerationParams, generate_sample
from src.presets import DRAM_PRESET_NAMES, FINFET_PRESET_NAMES


# =============================================================================
# On-The-Fly Dataset
# =============================================================================

class DriftSenseOnTheFlyDataset(Dataset):
    """
    Generates training triplets on-the-fly using the physics-based pipeline.
    Each triplet: (anchor, positive, negative) where:
      - anchor = downsampled reference (100x100)
      - positive = search crop at TRUE location (100x100)
      - negative = search crop at WRONG periodic location (100x100)
    """

    def __init__(self, num_triplets=4096, arch_subset=None):
        self.num_triplets = num_triplets
        if arch_subset is None:
            self.presets = DRAM_PRESET_NAMES + FINFET_PRESET_NAMES
        else:
            self.presets = arch_subset
        self.params = GenerationParams()

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # Use a deterministic seed derived from idx for reproducibility
        seed = (idx * 7919 + 12345) % (2**31)
        rng = np.random.default_rng(seed)

        # Pick random preset
        arch = rng.choice(self.presets)

        # Generate sample using the physics pipeline
        sample = generate_sample(arch, rng, self.params)

        # CRITICAL FIX: dict access, not attribute access
        ref_img = sample['reference_img']      # 1000x1000 uint8
        search_img = sample['search_img']      # 1000x1000 uint8
        x0, y0, w, h = sample['gt_box']        # in search coordinates (x0, y0, ~100, ~100)

        x0, y0, w, h = int(x0), int(y0), int(w), int(h)

        # --- ANCHOR: Downsample reference_img by 10x ---
        # Use AREA downsampling to match pipeline's downsample_area_average
        ref_pil = Image.fromarray(ref_img)
        anchor_pil = ref_pil.resize((ref_img.shape[1] // 10, ref_img.shape[0] // 10), Image.Resampling.LANCZOS)
        anchor = np.array(anchor_pil, dtype=np.float32) / 255.0

        # --- POSITIVE: Crop search_img at gt_box ---
        x0_c = max(0, min(x0, search_img.shape[1] - w))
        y0_c = max(0, min(y0, search_img.shape[0] - h))
        positive = search_img[y0_c:y0_c+h, x0_c:x0_c+w].astype(np.float32) / 255.0

        # --- NEGATIVE: Crop search_img at WRONG periodic-like location ---
        # CRITICAL FIX: Generate negatives at offsets that mimic real ambiguity.
        # The search image has repeating patterns. A wrong match could be:
        #   (a) A periodic offset (e.g., shifted by one pattern pitch)
        #   (b) A random location far from the true location
        # We mix both for robust training.
        neg_success = False
        neg_attempts = 0
        while not neg_success and neg_attempts < 50:
            neg_attempts += 1

            # 60% chance: periodic offset (the real source of ambiguity)
            if rng.random() < 0.6:
                # Pick a direction and step size (mimicking pattern pitch ~30-100px)
                pitch = rng.integers(30, 100)
                dx = rng.choice([-2, -1, 1, 2]) * pitch
                dy = rng.choice([-2, -1, 1, 2]) * pitch
                nx0 = x0_c + dx
                ny0 = y0_c + dy
            else:
                # 40% chance: random location anywhere
                nx0 = rng.integers(0, max(1, search_img.shape[1] - w))
                ny0 = rng.integers(0, max(1, search_img.shape[0] - h))

            # Ensure negative is valid and not too close to true location
            nx0 = max(0, min(nx0, search_img.shape[1] - w))
            ny0 = max(0, min(ny0, search_img.shape[0] - h))

            if abs(nx0 - x0_c) >= 30 or abs(ny0 - y0_c) >= 30:
                negative = search_img[ny0:ny0+h, nx0:nx0+w].astype(np.float32) / 255.0
                neg_success = True

        if not neg_success:
            # Fallback: just pick a random location
            nx0 = rng.integers(0, max(1, search_img.shape[1] - w))
            ny0 = rng.integers(0, max(1, search_img.shape[0] - h))
            negative = search_img[ny0:ny0+h, nx0:nx0+w].astype(np.float32) / 255.0

        # --- Resize all to exactly 100x100 (belt and suspenders) ---
        anchor = np.array(Image.fromarray((anchor * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        positive = np.array(Image.fromarray((positive * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        negative = np.array(Image.fromarray((negative * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

        # Add channel dim: (1, H, W)
        anchor_t = torch.from_numpy(anchor).unsqueeze(0)
        positive_t = torch.from_numpy(positive).unsqueeze(0)
        negative_t = torch.from_numpy(negative).unsqueeze(0)

        return anchor_t, positive_t, negative_t


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, loader, device):
    """Compute triplet ranking accuracy: fraction where pos is closer than neg."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            out_a = model(a)
            out_p = model(p)
            out_n = model(n)

            # Embeddings are already L2 normalized by TinyEncoder
            dist_pos = torch.norm(out_a - out_p, p=2, dim=1)
            dist_neg = torch.norm(out_a - out_n, p=2, dim=1)

            correct += (dist_pos < dist_neg).sum().item()
            total += a.size(0)

    return correct / total if total > 0 else 0


# =============================================================================
# Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Drift-Sense Siamese Ranker")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--margin', type=float, default=0.3, help="Triplet margin")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-train', type=int, default=4096, help="Training samples")
    parser.add_argument('--num-val', type=int, default=512, help="Validation samples")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training config: epochs={args.epochs}, lr={args.lr}, margin={args.margin}, batch={args.batch_size}")

    os.makedirs('weights', exist_ok=True)

    model = SiameseRanker().to(device)

    train_dataset = DriftSenseOnTheFlyDataset(num_triplets=args.num_train)
    val_dataset = DriftSenseOnTheFlyDataset(num_triplets=args.num_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    warmup_epochs = 3

    for epoch in range(args.epochs):
        # Manual warmup
        if epoch < warmup_epochs:
            lr = args.lr * ((epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        train_loss = 0.0
        num_batches = 0

        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()

            out_a = model(a)
            out_p = model(p)
            out_n = model(n)

            loss = criterion(out_a, out_p, out_n)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * a.size(0)
            num_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        train_loss /= len(train_dataset)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights/siamese_ranker.pth')
            patience_counter = 0
            print(f"  [*] Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and export to ONNX
    if os.path.exists('weights/siamese_ranker.pth'):
        model.load_state_dict(torch.load('weights/siamese_ranker.pth', map_location=device))
        print(f"Loaded best model (val_acc={best_val_acc:.4f})")

    export_to_onnx(model, 'weights/siamese_ranker.onnx')
    print("Exported model to weights/siamese_ranker.onnx")


if __name__ == '__main__':
    main()
