# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

> **SEMICON India 2026 Hackathon — Applied Materials Challenge**

## Overview

Drift-Sense solves the Navigation-Error Recovery problem in semiconductor wafer inspection: finding the exact location of a high-resolution reference pattern inside a lower-resolution, noisy search image captured by a scanning electron microscope (SEM).

Our approach combines **classical computer vision** (multi-scale normalized cross-correlation with gradient fusion) with an **AI-powered Siamese neural network** that disambiguates periodic structures by learning unique local defect fingerprints.

### Key Features
- **Multi-scale template matching** with dual-domain NCC (intensity + gradient)
- **Siamese CNN candidate ranker** for periodic-ambiguity resolution
- **Sub-pixel accuracy** via phase correlation refinement
- **Graceful fallback**: ONNX → PyTorch → Classical (never crashes)
- **Physically-grounded synthetic data** with 13 SEM degradation effects

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
```bash
# Generate 30 DRAM + 30 FinFET pairs (60 total)
python generate_dataset.py --architecture both --num-pairs 30 --output dataset --seed 42

# Generate DRAM-only
python generate_dataset.py --architecture dram --num-pairs 30 --output dataset

# Generate FinFET-only  
python generate_dataset.py --architecture finfet --num-pairs 30 --output dataset
```

### 3. Run Inference (Single Pair)
```bash
python localize.py --reference dataset/pair_0000/reference.png --search dataset/pair_0000/search.png
```

Output format: `x.xx,y.yy` (predicted center coordinates in search image)

### 4. Run Benchmark
```bash
python benchmark.py --dataset dataset
```

### 5. Train the Siamese Network (Optional, requires GPU)
```bash
python train_siamese.py --epochs 20
```

Trained weights are saved to `weights/siamese_ranker.pth` and `weights/siamese_ranker.onnx`.

## Project Structure

```
Semicon1/
├── localize.py              # ⭐ Main inference script (Applied Materials runs this)
├── generate_dataset.py      # Synthetic dataset generator
├── train_siamese.py         # Siamese network training script
├── siamese_net.py           # Siamese CNN architecture
├── benchmark.py             # Three-way accuracy benchmark
├── evaluate.py              # Single-pair evaluation
├── citations.md             # Academic references for all augmentations
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── weights/                 # Trained model weights
│   ├── siamese_ranker.pth   # PyTorch weights
│   └── siamese_ranker.onnx  # ONNX export
└── src/                     # Core pipeline modules
    ├── pipeline.py          # Dataset generation orchestrator
    ├── sem_imaging.py       # SEM physics simulation
    ├── presets.py           # DRAM/FinFET structural presets
    ├── structural_defects.py # Pattern collapse modeling
    └── patterns/
        ├── dram.py          # DRAM cell array generator
        ├── finfet.py        # FinFET gate array generator
        └── zones.py         # Mat/strip zone composition
```

## Dataset Format

- **Reference Image**: 1000×1000 px @ 1 nm/px (1 μm field of view)
- **Search Image**: 1000×1000 px @ 10 nm/px (10 μm field of view)
- The reference pattern appears as a ~100×100 px region inside the search image
- Ground truth: center (x, y) coordinates of the matching region in search image pixels

## Algorithm Pipeline

1. **Multi-Scale Downsample**: Reference is downsampled by factors [9.0, 9.5, 10.0, 10.5, 11.0]
2. **Dual-Domain NCC**: Intensity + gradient normalized cross-correlation, fused
3. **Peak Extraction**: Local maxima detection with periodic-aware footprint
4. **AI Disambiguation**: Siamese CNN ranks ambiguous candidates (if weights available)
5. **Center Heuristic**: Fallback to closest-to-center for periodic patterns
6. **Subpixel Refinement**: Phase correlation for sub-pixel accuracy

## Technology Stack

- Python 3.10+
- OpenCV (template matching, image processing)
- NumPy (numerical computation)
- PyTorch (Siamese network training/inference)
- ONNX Runtime (fast inference without PyTorch)
- scikit-image (phase cross-correlation)
- SciPy (peak detection, statistics)

## Hardware

- Development: Windows 11, NVIDIA RTX 3050 6GB
- Inference: CPU-only compatible (GPU optional)

## Team

**VisionForge** — SEMICON India 2026 Hackathon

## References

See [citations.md](citations.md) for complete academic references justifying all augmentation and noise model choices.
