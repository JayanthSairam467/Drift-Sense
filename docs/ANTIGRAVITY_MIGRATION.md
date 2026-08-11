# Antigravity to OpenCode Migration Context

This document captures the essential history, context, and debugging discoveries from the previous Antigravity AI sessions. It is the bridge for OpenCode agents to understand *why* the code is written the way it is.

## 1. Project Background
- **Goal:** Win the Applied Materials "Drift-Sense" challenge at the SEMICON India 2026 Hackathon.
- **Hardware Constraint:** The primary dev machine has an RTX 3050 (6GB). We experienced severe CPU bottlenecks when generating training data on-the-fly, which spawned the offline training strategy.
- **Evaluation Constraint:** Hackathon judges will run this script on fresh machines. The inference time and model size are explicitly graded. Heavy transformer models (DINOv2, Swin) were abandoned because they violate this constraint and fail to run efficiently on CPUs.

## 2. Debugging Discoveries (The Hard Lessons)

### The "Center Heuristic" Disaster
- **Initial state:** `localize.py` used to pick the NCC peak closest to the center of the image when multiple periodic peaks were found.
- **Discovery:** This actively ruined accuracy. Sometimes the true peak was at the edge, but the center heuristic forced the algorithm to pick a worse peak near the center. 
- **Fix:** We completely removed the center heuristic. We now *always* trust the top NCC peak (after cross-scale voting). This instantly restored Classical accuracy to match Baseline.

### The AI "Overrider" Disaster
- **Initial state:** The Siamese AI was allowed to unilaterally override the Classical NCC result if it felt confident.
- **Discovery:** The AI model, when poorly trained, was constantly overriding *correct* classical predictions with *wrong* ones, bringing accuracy down to 6%.
- **Fix:** The AI is now strictly a **Tie-Breaker**. It only activates if the ratio between the top two Classical NCC peaks is < 1.03 (meaning NCC is confused). Furthermore, the AI is restricted to only picking among the top-5 Classical NCC candidates. **Rule: AI can only help, never hurt.**

### The Unsolvable Periodic Problem & The "Structural Defect" Breakthrough
- **The Problem:** We found that on highly periodic patterns (FinFET), the Classical and AI hybrid models both failed on a few specific images with 500px+ errors.
- **The Root Cause:** We discovered that in the dataset generator, independent noise was applied to perfectly periodic mathematical grids. This meant that at the Ground Truth location, the NCC score was genuinely 15% *lower* than at a random false periodic location. The true match physically looked worse than a false match.
- **The Fix:** We injected **Structural Defects** (missing contacts, oxide particles, micro-scratches) into the fine canvas *before* taking the crop and applying SEM imaging. This ensures that the unique defects appear in *both* the reference and search images. This breaks the periodicity and gives the AI actual unique fingerprints to learn. 

### The Missing Noise Bug
- **The Problem:** The difficulty flags (`easy`, `hard`, `extreme`) were having no effect on accuracy in the benchmark.
- **The Root Cause:** The `generate_dataset.py` script was passing kwargs like `detector_noise` and `speckle` instead of the correct `GenerationParams` attributes like `detector_noise_sigma_search` and `speckle_sigma`. The pipeline silently ignored them.
- **The Fix:** We updated `generate_dataset.py` to use the correct variable names, making the difficult datasets actually difficult.

## 3. Important Constraints for Future Development
- **Do not refactor the classical NCC pipeline to rely completely on deep learning.** The classical pipeline is incredibly fast and highly accurate for 85% of cases.
- **Preserve the `benchmark.py` testing flow.** Always ensure `CLASSICAL >= BASELINE` and `AI_HYBRID >= CLASSICAL`.
- **The user explicitly stated:** "I am not satisfied with the accuracy of the code... reach 100% in all possible ways... correct the code according to it." and "We can't afford a single issue because the deadline is almost two days ahead".

## 4. Known Discrepancies between Antigravity History and Current Code
- Previous conversations discussed adding a `train_siamese.py` script for on-the-fly generation. This has been fully deprecated in favor of `generate_triplets.py` and `train_offline.py`.
- The first implementation plan (still in the repo as `implementation_plan semicon 1.md`) mentions using a `center heuristic`. That document is now outdated; the code is the source of truth, and the center heuristic has been aggressively removed.

## 5. User Preferences
- The user has limited time, cannot easily use `git` (admin rights issue), and has a lower-end GPU.
- The user wants precise, fully-tested code without assumptions. 
- "So, imagine yourself as a master—an AI-ML developer master—and make this code so that we won't be in any trouble or get overwhelmed by this project. Be 100% sure about your work."
