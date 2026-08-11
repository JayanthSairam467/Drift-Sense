# Drift-Sense: Current State

## COMPLETED
- `src/pipeline.py` and `src/sem_imaging.py`: Fully functional, physically grounded SEM synthetic data generation.
- `localize.py`: v5.0 Final architecture deployed. Features dual-domain NCC, 9-scale cross-consistency voting, Siamese CNN top-5 candidate tiebreaker, and phase cross-correlation subpixel refinement.
- `siamese_net.py`: Lightweight `TinyEncoder` (under 300k params) built in PyTorch with ONNX export capability.
- `generate_dataset.py`: Fixed to inject structural defects (`missing_contact`, `bright_particle`, `scratch`) into the canvas *before* imaging, providing the necessary signal for AI disambiguation.
- `benchmark.py`: Evaluator comparing Baseline vs Classical vs AI Hybrid.
- `generate_triplets.py`: Pre-generates anchor/positive/negative datasets for offline training, bypassing CPU bottlenecks on slow hardware.

## IN PROGRESS
- Training the final AI model. The code to train it (`train_offline.py`) is complete, and the data generator (`generate_triplets.py`) is complete.

## NOT STARTED
- Running the massive triplet generation (e.g. 10,000+ triplets) and executing the final offline training loop to get the absolute best `.pth` and `.onnx` weights.
- Building the final Hackathon presentation (PPT/PDF).

## KNOWN BUGS
- None currently known. Previous bug (difficulty settings ignoring noise variables in `generate_dataset.py`) was fixed. Previous bug (AI overriding correct classical predictions) was fixed by restricting AI to tiebreaker status only (confidence < 1.03 ratio).

## KNOWN TECHNICAL DEBT
- The ONNX fallback in `localize.py` is present but the primary load mechanism relies on `siamese_net.py` and PyTorch. If PyTorch is unavailable on the judging machine, `localize.py` falls back to Classical gracefully, but ONNX could be wired up as a primary AI fallback to ensure the AI still runs without PyTorch.

## BLOCKERS
- None.

## NEXT PRIORITIES
1. Generate the large training dataset via `python generate_triplets.py --num-triplets 5000 --output training_data`.
2. Train the Siamese network via `python train_offline.py --data-dir training_data --epochs 20`.
3. Re-run `benchmark.py` to prove that `AI_HYBRID` now beats `CLASSICAL` on periodic, ambiguous FinFET patterns thanks to the newly added structural defects.
