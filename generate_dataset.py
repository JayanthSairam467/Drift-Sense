"""
Generate Wafer Inspection Dataset for Drift-Sense Hackathon.

Uses the official src/pipeline.py reference implementation with correct
GenerationParams field names and structural defect injection for
periodic-pattern disambiguation.
"""

import os
import json
import argparse
import random
import csv
import cv2
import numpy as np
from PIL import Image

from src.pipeline import (
    GenerationParams, generate_fine_canvas_zoned, REFERENCE_SIZE_PX,
    SCALE_FACTOR, PIXEL_SIZE_REF_NM, PIXEL_SIZE_SEARCH_NM,
    FINE_CANVAS_SIZE_PX, compute_gt,
)
from src import sem_imaging
from src.presets import DRAM_PRESET_NAMES, FINFET_PRESET_NAMES

DIFFICULTIES = ['easy', 'medium', 'hard', 'extreme']
DIFFICULTY_WEIGHTS = [0.3, 0.4, 0.2, 0.1]


def get_difficulty_params(difficulty):
    """Return GenerationParams with CORRECT field names for each difficulty."""
    params = GenerationParams()
    if difficulty == 'easy':
        params.dose_search = 800.0
        params.detector_noise_sigma_search = 2.0
        params.shear_amplitude_px = 0.5
        params.drift_jitter_px = 0.2
    elif difficulty == 'medium':
        params.dose_search = 200.0
        params.detector_noise_sigma_search = 5.0
        params.shear_amplitude_px = 1.5
        params.drift_jitter_px = 0.5
    elif difficulty == 'hard':
        params.dose_search = 60.0
        params.detector_noise_sigma_search = 8.0
        params.speckle_sigma = 0.15
        params.shear_amplitude_px = 2.5
        params.drift_jitter_px = 1.0
    elif difficulty == 'extreme':
        params.dose_search = 25.0
        params.detector_noise_sigma_search = 12.0
        params.speckle_sigma = 0.3
        params.salt_pepper_prob = 0.01
        params.shear_amplitude_px = 4.0
        params.drift_jitter_px = 2.0
    return params


def add_structural_defects(canvas, rng, num_defects=None):
    """
    Add permanent structural defects to the fine canvas (10000x10000, 1nm/px).

    These simulate real manufacturing imperfections: missing contacts, oxide
    particles, and micro-scratches. Because they are added to the CANVAS
    before any SEM imaging, they appear identically in both reference and
    search images, creating unique spatial fingerprints that break the
    periodicity of the underlying pattern.

    This is the KEY to disambiguation: at the true match location, both
    images share the same defect pattern; at periodic offsets, defects
    differ (or are absent). Template matching (NCC + gradient) and the
    Siamese AI can exploit this.
    """
    h, w = canvas.shape[:2]
    if num_defects is None:
        num_defects = rng.integers(5, 15)

    for _ in range(num_defects):
        defect_type = rng.choice(['missing_contact', 'bright_particle', 'scratch'])
        # Place defects randomly across the canvas
        x = int(rng.integers(100, w - 100))
        y = int(rng.integers(100, h - 100))

        if defect_type == 'missing_contact':
            # Dark void: 15-40 nm radius (in 1nm/px canvas = 15-40 pixels)
            r = int(rng.integers(15, 40))
            cv2.circle(canvas, (x, y), r, 0.0, -1)
        elif defect_type == 'bright_particle':
            # Bright oxide particle: 10-25 nm radius
            r = int(rng.integers(10, 25))
            brightness = float(rng.uniform(200, 255))
            cv2.circle(canvas, (x, y), r, brightness, -1)
        elif defect_type == 'scratch':
            # Micro-scratch: 50-200 nm long, thin line
            length = int(rng.integers(50, 200))
            angle = float(rng.uniform(0, np.pi))
            x2 = int(x + length * np.cos(angle))
            y2 = int(y + length * np.sin(angle))
            cv2.line(canvas, (x, y), (x2, y2), float(rng.uniform(30, 100)), 2)

    return canvas


def generate_sample_with_defects(preset, rng, params):
    """
    Generate a sample using the pipeline but with structural defects
    injected into the fine canvas BEFORE SEM imaging.
    """
    # Step 1: Generate the fine canvas (pattern only, no imaging)
    zone_result = generate_fine_canvas_zoned(preset, rng, params)
    fine_canvas = zone_result["canvas"]

    # Step 2: Inject structural defects into the canvas
    fine_canvas = add_structural_defects(fine_canvas, rng)

    # Step 3: Pick crop location for reference (pulled inward when the search
    # image is rotated/zoomed, so the reference box stays fully in-frame).
    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    rot_margin = int(
        REFERENCE_SIZE_PX // SCALE_FACTOR
        if params.rotation_deg != 0.0 or params.scale_search != 1.0
        else 0
    )
    lo = rot_margin * SCALE_FACTOR
    hi = max_offset - rot_margin * SCALE_FACTOR
    strip_rects = zone_result.get("strip_rects") or []

    if strip_rects and rng.random() < params.boundary_bias:
        sr = strip_rects[int(rng.integers(0, len(strip_rects)))]
        sx, sy, sw, sh = sr
        scx, scy = sx + sw / 2.0, sy + sh / 2.0
        x0 = scx - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        y0 = scy - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        x0 = int(np.clip(x0, lo, hi))
        y0 = int(np.clip(y0, lo, hi))
    else:
        x0 = int(rng.integers(lo, hi + 1))
        y0 = int(rng.integers(lo, hi + 1))

    crop = fine_canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    # Step 4: SEM imaging (independent noise for ref and search)
    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        edge_brightening_strength=params.edge_brightening_strength,
    )

    search_img = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        rotation_deg=params.rotation_deg,
        scale_search=params.scale_search,
        edge_brightening_strength=params.edge_brightening_strength,
    )

    # Step 5: Ground truth (affine-transformed when the search image was
    # rotated/zoomed, so GT marks the true location in the final image).
    gt_x0, gt_y0, box_w, box_h = compute_gt(x0, y0, params)
    gt_cx = gt_x0 + box_w / 2.0
    gt_cy = gt_y0 + box_h / 2.0

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "architecture": preset,
        "params": params.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Wafer Inspection Dataset")
    parser.add_argument('--architecture', choices=['dram', 'finfet', 'both'],
                        default='both', help="Architecture family")
    parser.add_argument('--num-pairs', type=int, default=30,
                        help="Number of pairs to generate")
    parser.add_argument('--output', type=str, default='dataset',
                        help="Output directory")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    manifest_path = os.path.join(args.output, 'manifest.csv')
    manifest_file = open(manifest_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(manifest_file)
    csv_writer.writerow([
        'pair_id', 'reference_path', 'search_path',
        'gt_x', 'gt_y', 'architecture', 'difficulty'
    ])

    for i in range(args.num_pairs):
        arch_family = args.architecture
        if arch_family == 'both':
            arch_family = rng.choice(['dram', 'finfet'])

        if arch_family == 'dram':
            preset = rng.choice(DRAM_PRESET_NAMES)
        else:
            preset = rng.choice(FINFET_PRESET_NAMES)

        difficulty = rng.choice(DIFFICULTIES, p=DIFFICULTY_WEIGHTS)
        params = get_difficulty_params(difficulty)

        # Randomize the new physics effects per-sample: edge brightening is a
        # universal SEM behaviour (appears always, mildly); scan-field
        # rotation and magnification calibration drift are acquisition
        # conditions that the tool sees on some frames. GT is transformed by
        # the exact same affine parameters inside generate_sample_with_defects.
        params.edge_brightening_strength = float(rng.uniform(0.2, 0.6))
        if rng.random() < 0.5:
            params.rotation_deg = float(rng.uniform(-2.0, 2.0))
            params.scale_search = float(rng.uniform(0.98, 1.02))

        sample = generate_sample_with_defects(preset, rng, params)

        pair_dir_name = f'pair_{i:04d}'
        pair_dir = os.path.join(args.output, pair_dir_name)
        os.makedirs(pair_dir, exist_ok=True)

        ref_path = os.path.join(pair_dir, 'reference.png')
        search_path = os.path.join(pair_dir, 'search.png')
        gt_path = os.path.join(pair_dir, 'ground_truth.json')

        ref_img = Image.fromarray(sample['reference_img'].astype(np.uint8), mode='L')
        search_img = Image.fromarray(sample['search_img'].astype(np.uint8), mode='L')
        ref_img.save(ref_path)
        search_img.save(search_path)

        gt_data = {
            'center_x': float(sample['gt_x']),
            'center_y': float(sample['gt_y']),
            'gt_box': [float(x) for x in sample['gt_box']],
            'architecture': preset,
            'difficulty': difficulty
        }

        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=4)

        ref_rel = os.path.join(pair_dir_name, 'reference.png').replace('\\', '/')
        search_rel = os.path.join(pair_dir_name, 'search.png').replace('\\', '/')

        csv_writer.writerow([
            pair_dir_name, ref_rel, search_rel,
            gt_data['center_x'], gt_data['center_y'],
            preset, difficulty
        ])
        print(f"Generated pair {i+1}/{args.num_pairs}...")

    manifest_file.close()
    print(f"\nDataset saved to {args.output}")


if __name__ == '__main__':
    main()
