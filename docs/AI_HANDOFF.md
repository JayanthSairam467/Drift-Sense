# AI HANDOFF — READ THIS FIRST

## 1. What is this project?
Drift-Sense is a template-matching and localization pipeline for semiconductor wafer inspection (SEM images). The goal is finding a high-resolution reference pattern (1000x1000 @ 1nm/px) inside a lower-resolution, noisy search image (1000x1000 @ 10nm/px).

## 2. What is the architecture?
A **Classical-Smart Hybrid**:
- **Classical Core**: Dual-Domain NCC (Intensity + Gradient) across 9 scales, with cross-scale voting to rank peaks. Phase cross-correlation provides sub-pixel accuracy.
- **AI Tiebreaker**: A tiny 4-layer Siamese CNN (PyTorch) is invoked *only* when the top-2 classical peaks are within 3% of each other. It scores the top-5 candidates to break periodic ambiguity.

## 3. What is currently working?
- The synthetic dataset generator (`src/pipeline.py`, `generate_dataset.py`) is fully functional. Crucially, it now injects *structural defects* (scratches, oxide spots) into the images to break mathematical periodicity.
- The inference script (`localize.py`) is locked and highly optimized.
- Offline data generation (`generate_triplets.py`) and training (`train_offline.py`) scripts are complete.

## 4. What is currently broken?
Nothing is fundamentally broken, but the final, high-quality AI weights are not yet trained on the newly-fixed defect-injected dataset.

## 5. What was being worked on most recently?
The dataset generator (`generate_dataset.py` and `generate_triplets.py`) was just fixed to properly inject structural defects and utilize noise parameters correctly.

## 6. What should be done next?
1. Generate the training triplets: `python generate_triplets.py --num-triplets 5000 --output training_data`
2. Train the Siamese CNN: `python train_offline.py --data-dir training_data --epochs 20`
3. Generate a benchmark dataset: `python generate_dataset.py --architecture both --num-pairs 30 --output dataset`
4. Run the benchmark to verify the AI now helps: `python benchmark.py --dataset dataset`

## 7. What files should be inspected first?
- `AGENTS.md` - High level rules.
- `docs/ANTIGRAVITY_MIGRATION.md` - Critical history of debugging (why things are built this way).
- `localize.py` - The main algorithmic core.

## 8. What should the AI absolutely avoid changing?
- **DO NOT** change the tiebreaker logic in `localize.py` (AI only kicks in if confidence ratio < 1.03).
- **DO NOT** remove structural defect injection in `generate_dataset.py` (it is required for the AI to have any purpose).
- **DO NOT** rewrite the architecture to rely solely on deep learning (must remain a classical-first hybrid).

## 9. What commands should it run before modifying anything?
Run the benchmark to establish a baseline before you touch the algorithmic logic:
```bash
python benchmark.py --dataset dataset
```

## 10. What tests should it run after changes?
```bash
python benchmark.py --dataset dataset
```
Ensure `CLASSICAL >= BASELINE` and `AI_HYBRID >= CLASSICAL`. If either fails, revert your changes immediately.
