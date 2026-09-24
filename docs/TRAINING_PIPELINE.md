# Training Pipeline

## 1. Dataset Curation

The project began with FASDD data and applied dataset auditing, exact-duplicate analysis, visual duplicate analysis, near-duplicate detection, leakage analysis, sanitization, and validation.

The curated dataset lineage is referred to as `FASDD CLEAN V1`.

Relevant utilities are under `scripts/dataset_curation/`.

## 2. R1 — Architecture Comparison

R1 compared YOLOv8n, YOLOv8s, and YOLOv8m using the same R1 dataset and validation protocol.

YOLOv8s was selected as the downstream lineage after the architecture comparison.

Artifacts are stored under `results/r1/`.

## 3. Hard-Negative Mining

False-positive cases from previously unseen clean negative images were mined and manually reviewed.

The reviewed hard negatives were separated into training and held-out validation subsets.

Scripts are stored under `scripts/hard_negative/`.

## 4. R2

R2 initialized from the R1 YOLOv8s checkpoint and incorporated verified hard negatives.

The purpose was to reduce false positives while preserving useful positive-detection performance.

Artifacts are stored under `results/r2/`.

## 5. R2.1

R2.1 was a preserve-positive refinement initialized from R2.

Its objective was to recover positive sensitivity while retaining the hard-negative learning introduced in R2.

Artifacts are stored under `results/r21/`.

## 6. Internet Proxy Negative Evaluation

Additional proxy-negative imagery was used to evaluate false-alert behavior outside the original validation distribution.

Proxy imagery was split by camera identity to avoid overlap between training proxy cameras and held-out proxy cameras.

## 7. R3

R3 initialized from R2.1 and incorporated proxy hard negatives.

Multiple R3 checkpoints were evaluated using validation metrics together with held-out proxy-negative false-positive behavior.

Artifacts are stored under `results/r3/`.

## 8. Final Model Selection

Checkpoint `epoch6.pt` from R3 was selected before opening the final test set.

The selected checkpoint was frozen and recorded in `results/final_test/model_freeze/FINAL_MODEL_FREEZE.json`.

## 9. Locked Final Test

The final test dataset was opened only after model selection and model freeze, and the test run was performed once.

Records:

- `results/final_test/FINAL_TEST_COMPLETED.json`
- `results/final_test/FINAL_TEST_SUMMARY.txt`
- `metadata/datasets/final_test_TEST_LOCK.json`

The final test must not be reused for model tuning or checkpoint selection.

## 10. Deployment Export

The frozen PyTorch model was exported to ONNX.

PT-to-ONNX equivalence was evaluated using validation data, not final-test data.

Deployment artifacts are stored under `results/deployment/`.
