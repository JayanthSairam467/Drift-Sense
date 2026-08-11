# Drift-Sense: Architectural Decisions

## Decision 1: CPU-First Classical-Smart Hybrid Architecture
- **Context:** The hackathon explicitly grades inference time and model size, and requires code to run on organizers' machines. Heavy transformers (DINOv2, Swin) have massive domain gaps for grayscale SEM images and are too slow/heavy.
- **Chosen Approach:** Use classical CV (Normalized Cross-Correlation) as the primary engine. Use AI only as a secondary tie-breaker.
- **Why it was chosen:** Guarantees sub-second execution on a CPU. Guarantees that the AI cannot accidentally ruin a clear, correct classical prediction.
- **Consequences:** We cap the maximum potential AI harm to 0%, while retaining all the upside for highly ambiguous (periodic) images.

## Decision 2: Dual-Domain Normalized Cross-Correlation (NCC)
- **Context:** Standard NCC on pixel intensities can fail when image contrast fluctuates or charging effects appear.
- **Chosen Approach:** Run NCC on raw intensity (70% weight) AND Sobel gradient magnitude (30% weight).
- **Why it was chosen:** Gradients are invariant to global illumination/brightness shifts, which are common in SEM due to dose variations and charging effects.
- **Consequences:** More robust peak finding in noisy conditions.

## Decision 3: Structural Defect Injection in Data Generator
- **Context:** In perfectly periodic structures (like FinFET arrays), independent random noise applied to the reference and search images causes random false periodic locations to have higher NCC scores than the true ground-truth location. No algorithm can solve this if the true location has genuinely lower signal.
- **Chosen Approach:** Inject structural defects (`missing_contact`, `bright_particle`, `scratch`) into the 10000x10000 fine canvas *before* taking the reference crop and applying SEM degradations.
- **Why it was chosen:** It forces unique spatial "fingerprints" to exist at the ground truth location. The Siamese network can learn to look for these fingerprints to disambiguate identical-looking periodic arrays.
- **Consequences:** The dataset generator accurately reflects reality (wafers have defects) and makes the problem actually solvable by AI.

## Decision 4: Cross-Scale Consistency Voting
- **Context:** Instead of just relying on the exact 10x downsample, scale variations can alter which peak is mathematically highest due to sub-pixel aliasing.
- **Chosen Approach:** Sweep scales from 9.0x to 11.0x (9 steps). Count how many times a candidate peak appears in the top-3 across all scales.
- **Why it was chosen:** A true peak is usually robust across small scale changes; a false peak caused by random noise correlation usually disappears if the scale changes slightly.
- **Consequences:** More accurate classical baseline before AI is even invoked.

## Decision 5: Subpixel Phase Cross-Correlation
- **Context:** The competition requires high accuracy. Standard template matching only yields integer pixel coordinates.
- **Chosen Approach:** Use `skimage.registration.phase_cross_correlation` with 100x upsampling on the final matched patch.
- **Why it was chosen:** It is computationally cheap (FFT based) and yields highly accurate sub-pixel shifts without needing sub-pixel sliding windows.
- **Consequences:** Sub-pixel accuracy down to 0.01px is achievable.

## Decision 6: Triplet Margin Loss for Offline Training
- **Context:** Need to train the Siamese Ranker efficiently.
- **Chosen Approach:** Generate (Anchor, Positive, Negative) triplets to disk via `generate_triplets.py`, then train via `train_offline.py` using `torch.nn.MarginRankingLoss`.
- **Why it was chosen:** Training on the fly (CPU data generation while GPU waits) was too slow on a laptop RTX 3050. Decoupling data generation from training solved the bottleneck.
- **Consequences:** Requires disk space for training data, but training completes in minutes instead of hours.
