# AGENTS.md

## Project Purpose
**Drift-Sense** is a solution for the Navigation-Error Recovery problem in semiconductor wafer inspection (specifically for the SEMICON India 2026 Hackathon — Applied Materials Challenge). The goal is to locate the exact center of a high-resolution reference pattern (1000x1000 px @ 1 nm/px) inside a lower-resolution, noisy search image (1000x1000 px @ 10 nm/px) captured by a scanning electron microscope (SEM).

## High-Level Architecture
The architecture is a **Classical-Smart Hybrid**. It explicitly avoids massive pretrained transformers (which are too heavy and have a large domain gap for grayscale SEM images).
1. **Classical Base:** Multi-scale dual-domain Normalized Cross-Correlation (NCC) using both intensity and gradient magnitude.
2. **AI Disambiguation:** A lightweight, custom Siamese CNN ranker is used *only* as a tie-breaker when classical NCC peaks are dangerously close (periodic ambiguity).
3. **Subpixel Refinement:** Phase cross-correlation (skimage) brings the output to subpixel accuracy.
4. **Data Engine:** Synthetic data generation pipeline (DRAM/FinFET) including physically-grounded SEM noise and, crucially, structural defects (e.g. missing contacts) to break periodicity and create unique fingerprints.

## Repository Structure
- `localize.py` - Core inference script. Returns (x, y) coordinates.
- `benchmark.py` - Three-way evaluator (Baseline vs. Classical vs. AI Hybrid).
- `generate_dataset.py` - Synthetic dataset generator (invokes pipeline).
- `generate_triplets.py` - Pre-generates anchor/positive/negative datasets for offline AI training.
- `train_offline.py` & `train_siamese.py` - AI training scripts.
- `siamese_net.py` - PyTorch definition of the TinyEncoder SiameseRanker.
- `src/` - Core domain modules:
  - `pipeline.py` - Data orchestrator.
  - `sem_imaging.py` - SEM physics simulation.
  - `presets.py` - Pattern configurations.
  - `patterns/` - Base canvas generators (DRAM, FinFET, zones).
- `dataset/`, `training_data/` - Output directories (can be regenerated).
- `weights/` - Contains PyTorch `.pth` and ONNX weights.
- `docs/` - Extensive technical project context and history.

## Technology Stack
- **Core:** Python 3.10+
- **Vision:** OpenCV, scikit-image, SciPy
- **AI/ML:** PyTorch, ONNX Runtime
- **Matrix/Math:** NumPy

## Coding Conventions
- Prefer standard scientific computing libraries (cv2, numpy, scipy, skimage).
- Avoid unnecessary external dependencies.
- Make scripts self-contained CLI tools (using `argparse`).
- Maintain Python docstrings explaining *why* things are done (physics logic is important).

## Important Architectural Rules
- **No Transformer/Heavy Models:** Do not use DINOv2, Swin, or other heavy pre-trained ImageNet models.
- **CPU-First Design:** The inference pipeline (`localize.py`) must run gracefully on CPU and without crashing, falling back to pure classical NCC if the AI model is missing or fails.
- **The Physics Justification:** Every augmentation or change in the dataset generator must be physically justified as resembling real semiconductor inspection phenomena.

## Commands
### Setup
```bash
pip install -r requirements.txt
```
### Generate Data
```bash
python generate_dataset.py --architecture both --num-pairs 30 --output dataset --seed 42
```
### Benchmark
```bash
python benchmark.py --dataset dataset
```
### Run Inference (single pair)
```bash
python localize.py --reference <ref_path> --search <search_path>
```
### Train AI (Offline strategy)
```bash
python generate_triplets.py --num-triplets 4000 --output training_data
python train_offline.py --data-dir training_data --epochs 20
```

## Important Constraints & "Do Not Touch" Directives
- **DO NOT change `localize.py`'s classical tiebreaker logic easily.** The rule `Classical >= Baseline` and `AI_HYBRID >= Classical` is maintained carefully by utilizing AI *only* for the top-5 ambiguous peaks.
- **DO NOT remove structural defects** from `generate_dataset.py`. The addition of `missing_contact`, `bright_particle`, and `scratch` to the fine canvas *before* SEM imaging is the only way the AI can disambiguate periodic structures.
- **DO NOT assume the AI is fully replacing NCC.** The AI acts strictly as a Candidate Ranker.

## Current Implementation Status
The core pipeline (`localize.py`), AI architecture, and defect-injected synthetic data generator are fully functional. The recent fix was in the dataset generator, ensuring structural defects actually appear in both reference and search images to enable AI learning.

## Next OpenCode Agent Tasks
Read `docs/AI_HANDOFF.md` for exact next steps.
