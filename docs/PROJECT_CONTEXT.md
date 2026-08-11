# Drift-Sense: Project Context

## A. What the Project Does
Drift-Sense is an algorithm and software pipeline designed to locate a specific, high-resolution "Reference" image patch (1000x1000 pixels at 1 nm/px) inside a larger, noisier, lower-resolution "Search" image (1000x1000 pixels at 10 nm/px). It simulates fixing navigation drift errors in scanning electron microscopes (SEM) during semiconductor wafer inspection.

## B. Why it Exists
This project was built for the **SEMICON India 2026 Hackathon — Applied Materials Challenge**. The challenge requires identifying the exact target coordinate within a drifted SEM image, dealing with heavy periodic patterns (like DRAM or FinFET) which cause naive template matching to fail by matching the wrong period.

## C. Users / Use Cases
- Hackathon Judges (evaluating the `localize.py` script for accuracy, speed, and CPU efficiency).
- Semiconductor inspection engineers who need robust coordinate registration across changing SEM zoom levels and high noise.

## D. Complete Architecture
The system uses a **Classical-Smart Hybrid Architecture**.
1. **Classical Stage:** Normalizes scale, runs dual-domain (Intensity + Gradient) Normalized Cross-Correlation (NCC) across a small scale sweep (9.0x to 11.0x). Evaluates cross-scale consistency to rank peaks.
2. **AI Stage:** If the top two classical NCC peaks are dangerously close in score (ambiguity), it uses a lightweight Siamese CNN (`SiameseRanker`) to score the top-5 candidates.
3. **Refinement Stage:** Applies subpixel phase cross-correlation on the final chosen peak to get highly accurate sub-pixel coordinates.
4. **Data Generation Pipeline:** A synthetic physics-grounded pipeline that creates perfectly aligned reference and search images injected with specific SEM noises and unique structural defects.

## E. Component-by-Component Explanation
- `localize.py`: The orchestrator of inference. It takes two paths, executes NCC, extracts peaks, calls the AI if needed, and refines the final location.
- `siamese_net.py`: Contains `TinyEncoder`, a small 4-layer CNN producing a 128-dim embedding. Uses cosine similarity to rank candidate patches against the reference.
- `src/pipeline.py` & `src/sem_imaging.py`: The physics engine. Turns mathematical grids (DRAM/FinFET) into realistic noisy SEM images using models for dose, blur, charging, edge brightening, and speckle noise.
- `generate_dataset.py`: Drives the pipeline to generate a benchmark evaluation dataset. Crucially injects structural defects (scratches, oxide) into the master canvas.
- `generate_triplets.py` & `train_offline.py`: Handles massive offline generation of Anchor/Positive/Negative triplets, allowing the GPU to train rapidly without CPU generation bottlenecks.
- `benchmark.py`: Evaluates the Baseline (simple 10x NCC), Classical (Dual-domain NCC + voting), and AI Hybrid (Classical + AI tiebreaker) to prove value.

## F. Data Flow
1. Generation: Math Patterns -> Add Defects -> `fine_canvas` (1nm/px) -> Crop -> SEM Degradation -> Output Reference & Search.
2. Inference: Reference & Search -> Downsample Ref (9-11x) -> Dual-Domain NCC -> Peak Extraction -> (Optional) Siamese Ranking -> Phase Correlation -> Final (x, y).
3. Training: Generate Triplet -> Model Embedding -> Triplet Margin Loss -> Backpropagation -> Save Weights.

## G. Request/Response Flow
N/A (CLI application, no web server). Input is file paths via argparse; output is a printed string `x.xx,y.yy` or JSON via stdout.

## H. Important Business Logic
- The **AI is explicitly restricted** to only act as a tiebreaker for the top 5 classical NCC candidates when confidence is low (ratio < 1.03). It is never allowed to override a clear classical winner. This ensures the AI can never hurt the baseline accuracy.
- Dual-Domain NCC uses 70% intensity and 30% gradient magnitude.

## I. AI/ML Architecture
- **Model:** Siamese CNN (`TinyEncoder`).
- **Input:** 1-channel Grayscale images, resized dynamically via AdaptiveAvgPool.
- **Output:** 128-dimensional L2-normalized embedding.
- **Loss:** Triplet Margin Loss (Anchor, Positive, Negative).
- **Parameters:** < 300K. Extremely lightweight.

## J. Database Architecture
N/A (File-system based dataset storage).

## K. Frontend Architecture
N/A.

## L. Backend Architecture
N/A (Local execution CLI scripts).

## M. External Services / APIs
None. Designed to run completely offline on an evaluator's local machine.

## N. Authentication / Security Model
None required.

## O. Deployment Architecture
Deployed as a set of Python scripts with a `requirements.txt`. Requires PyTorch, OpenCV, skimage. ONNX is supported as a fallback.

## P. Development Workflow
1. Develop algorithm tweaks in `localize.py`.
2. Generate small test set via `generate_dataset.py`.
3. Run `benchmark.py` to compare Baseline, Classical, and AI models.
4. Scale up training with `generate_triplets.py` and `train_offline.py`.

## Q. Current State of the Project
The architecture is locked and highly optimized. The dataset generator has just been fixed to properly include unique structural defects (so the AI actually has something to learn). The AI training pipeline is ready. 

## R. Known Limitations
- Pure periodic arrays without *any* structural defects are unsolvable by both classical and AI means due to independent random noise causing false peaks.

## S. Known Bugs
None currently active. Previous bugs regarding missing dataset parameters and AI architecture mismatches have been resolved.

## T. Future Work
- Finalizing the offline training run on a large triplet dataset to generate the ultimate `.pth` and `.onnx` weights for submission.
- Preparing the presentation/slide deck for the hackathon submission.
