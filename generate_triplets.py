"""
Drift-Sense: Pre-Generate Training Triplets to Disk
=====================================================
Step 1 of 2 for Offline Training.

Hard-negative mining: for every sample, the negatives are taken from the
ACTUAL top-NCC peaks in the search image (the candidates the localizer would
confuse with the true match on periodic patterns). This forces the Siamese
ranker to learn the defect-fingerprint that separates the true match from the
plausible periodic impostors -- the exact decision the deployed AI has to make.

Usage:
    python generate_triplets.py --num-triplets 4000 --output training_data
    python generate_triplets.py --num-triplets 500 --output training_data_val
"""

import argparse
import os
import json
import time

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import maximum_filter

from src.pipeline import GenerationParams
from src.presets import DRAM_PRESET_NAMES, FINFET_PRESET_NAMES
from generate_dataset import generate_sample_with_defects


def _top_peaks(fused, w, k=8):
    """Return (score, cx, cy) for the top-k local maxima of a fused NCC map."""
    fp = max(3, int(w * 0.15)); fp += 1 if fp % 2 == 0 else 0
    lm = maximum_filter(fused, size=fp)
    mask = (fused == lm) & (fused > 0.3 * fused.max())
    ys, xs = np.where(mask)
    scores = fused[ys, xs]
    order = np.argsort(scores)[::-1][:k]
    out = []
    for idx in order:
        cx = int(xs[idx]) + w // 2
        cy = int(ys[idx]) + w // 2
        out.append((float(scores[idx]), cx, cy))
    return out


def generate_triplets_from_sample(seed, presets, params, hard_negatives=True):
    """Generate several (anchor, positive, negative) triplets from ONE sample.

    Each sample only needs its 10000x10000 canvas rendered once; we then mine
    several distinct hard negatives from its real NCC-confusable peaks, so the
    cost per triplet drops ~3x. Returns a list of (anchor, positive, negative).
    """
    rng = np.random.default_rng(seed)

    arch = rng.choice(presets)
    params.rotation_deg = float(rng.uniform(-3.0, 3.0))
    params.scale_search = float(1.0 + rng.uniform(-0.03, 0.03))
    params.edge_brightening_strength = float(rng.uniform(0.15, 0.6))

    sample = generate_sample_with_defects(arch, rng, params)

    ref_img = sample['reference_img']
    search_img = sample['search_img']
    x0, y0, w, h = sample['gt_box']
    x0, y0, w, h = int(round(x0)), int(round(y0)), int(round(w)), int(round(h))

    anchor = np.array(Image.fromarray(ref_img).resize((100, 100), Image.LANCZOS))

    x0_c = max(0, min(x0, search_img.shape[1] - w))
    y0_c = max(0, min(y0, search_img.shape[0] - h))
    pos_crop = search_img[y0_c:y0_c+h, x0_c:x0_c+w]
    positive = np.array(Image.fromarray(pos_crop).resize((100, 100), Image.LANCZOS))

    triples = []

    if hard_negatives:
        sf = search_img.astype(np.float32)
        gr = cv2.Sobel(sf, cv2.CV_32F, 1, 0, ksize=3); gc = cv2.Sobel(sf, cv2.CV_32F, 0, 1, ksize=3)
        sg = np.sqrt(gr*gr + gc*gc); sg = sg / (sg.max() + 1e-9)
        t = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
        tg = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3); tgc = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
        tgr = np.sqrt(tg*tg + tgc*tgc); tgr = tgr / (tgr.max() + 1e-9)
        ni = cv2.matchTemplate(sf, t, cv2.TM_CCOEFF_NORMED)
        ng = cv2.matchTemplate(sg, tgr, cv2.TM_CCOEFF_NORMED)
        fused = 0.70 * ni + 0.30 * ng

        peaks = _top_peaks(fused, 100, k=18)
        peaks = [p for p in peaks if np.hypot(p[1] - (x0_c + w/2), p[2] - (y0_c + h/2)) > 25]

        if peaks:
            # produce up to 3 negatives from the most confusing peaks
            for picked in range(min(3, len(peaks))):
                _, px, py = peaks[picked]
                nx, ny = int(px - 50), int(py - 50)
                nx = max(0, min(nx, search_img.shape[1] - 100))
                ny = max(0, min(ny, search_img.shape[0] - 100))
                neg_crop = search_img[ny:ny+100, nx:nx+100]
                negative = np.array(Image.fromarray(neg_crop).resize((100, 100), Image.LANCZOS))
                triples.append((anchor, positive, negative))
        else:
            # fallback random offset
            nx = x0_c; ny = y0_c
            while abs(nx - x0_c) < 30 and abs(ny - y0_c) < 30:
                nx = x0_c + int(rng.integers(-300, 300))
                ny = y0_c + int(rng.integers(-300, 300))
                nx = max(0, min(nx, search_img.shape[1] - 100))
                ny = max(0, min(ny, search_img.shape[0] - 100))
            neg_crop = search_img[ny:ny+100, nx:nx+100]
            negative = np.array(Image.fromarray(neg_crop).resize((100, 100), Image.LANCZOS))
            triples.append((anchor, positive, negative))
    else:
        nx = x0_c; ny = y0_c
        while abs(nx - x0_c) < 30 and abs(ny - y0_c) < 30:
            nx = x0_c + int(rng.integers(-300, 300))
            ny = y0_c + int(rng.integers(-300, 300))
            nx = max(0, min(nx, search_img.shape[1] - 100))
            ny = max(0, min(ny, search_img.shape[0] - 100))
        neg_crop = search_img[ny:ny+100, nx:nx+100]
        negative = np.array(Image.fromarray(neg_crop).resize((100, 100), Image.LANCZOS))
        triples.append((anchor, positive, negative))

    return triples


def generate_triplet(idx, presets, params, hard_negatives=True):
    """Backwards-compatible wrapper: returns the first triplet of a sample."""
    triples = generate_triplets_from_sample(idx, presets, params, hard_negatives)
    return triples[0]


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate training triplets to disk"
    )
    parser.add_argument('--num-triplets', type=int, default=4000,
                        help="Number of triplets to generate")
    parser.add_argument('--output', type=str, default='training_data',
                        help="Output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--easy-negatives', action='store_true',
                        help="Use random offsets instead of hard-mined NCC peaks")
    args = parser.parse_args()

    presets = DRAM_PRESET_NAMES + FINFET_PRESET_NAMES
    params = GenerationParams()

    os.makedirs(args.output, exist_ok=True)
    anchor_dir = os.path.join(args.output, 'anchors')
    positive_dir = os.path.join(args.output, 'positives')
    negative_dir = os.path.join(args.output, 'negatives')
    os.makedirs(anchor_dir, exist_ok=True)
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    total = args.num_triplets
    start = time.time()
    written = 0
    sidx = 0

    while written < total:
        triples = generate_triplets_from_sample(
            sidx, presets, params, hard_negatives=(not args.easy_negatives)
        )
        sidx += 1
        for anchor, positive, negative in triples:
            if written >= total:
                break
            Image.fromarray(anchor, 'L').save(os.path.join(anchor_dir, f'{written:05d}.png'))
            Image.fromarray(positive, 'L').save(os.path.join(positive_dir, f'{written:05d}.png'))
            Image.fromarray(negative, 'L').save(os.path.join(negative_dir, f'{written:05d}.png'))
            written += 1

        if written % 10 == 0 or written == 0:
            elapsed = time.time() - start
            rate = written / elapsed if elapsed > 0 else 0
            remaining = (total - written) / rate if rate > 0 else 0
            print(f"  [{written:5d}/{total}] "
                  f"{rate:.1f} triplets/s | "
                  f"ETA: {remaining/60:.1f} min")

    elapsed = time.time() - start
    print(f"\nDone! Generated {total} triplets in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"Saved to: {os.path.abspath(args.output)}")

    meta = {
        'num_triplets': total,
        'seed': args.seed,
        'patch_size': 100,
        'channels': 1,
        'hard_negatives': (not args.easy_negatives),
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()