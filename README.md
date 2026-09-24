# Fire & Smoke Detection — R3-E6

Research and training reproducibility repository for a YOLOv8-based fire and smoke detection model.

This repository documents the model-development workflow from dataset curation through architecture comparison, hard-negative mining, model refinement, final model freeze, locked final testing, and deployment export.

## Final Model

- Architecture: YOLOv8s
- Classes: `0=Fire`, `1=Smoke`
- Input size: `768`
- Confidence threshold: `0.25`
- NMS IoU: `0.70`
- `max_det`: `300`
- Final checkpoint: R3 `epoch6.pt`

Frozen PyTorch artifact: `fire_smoke_r3_e6_final.pt`

SHA256: `49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

ONNX artifact: `fire_smoke_r3_e6_final_768_fp32.onnx`

SHA256: `e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

Model binaries are distributed separately from Git history.

## Development Pipeline

```text
FASDD source data
       ↓
Dataset audit / deduplication / leakage control
       ↓
FASDD CLEAN V1
       ↓
R1 architecture comparison: YOLOv8n / YOLOv8s / YOLOv8m
       ↓
YOLOv8s downstream lineage
       ↓
Hard-negative mining
       ↓
R2
       ↓
R2.1 preserve-positive refinement
       ↓
Internet proxy hard-negative evaluation
       ↓
R3
       ↓
Checkpoint comparison
       ↓
R3-E6 selected and frozen
       ↓
Locked Final Test
       ↓
ONNX export and validation
```

## Final Test

The final test set was locked and used only after final model selection and model freeze.

- Images: `15,863`
- Fire GT boxes: `9,005`
- Smoke GT boxes: `8,208`

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.8109 | 0.7364 | 0.8044 | 0.5256 |
| Fire | 0.8068 | 0.7815 | 0.8473 | 0.5382 |
| Smoke | 0.8150 | 0.6914 | 0.7615 | 0.5130 |

The final test was not reused for checkpoint selection, tuning, ONNX equivalence testing, or deployment optimization.

## Repository Structure

```text
configs/
docs/
metadata/
model/
results/
scripts/
```

See the documentation under `docs/` for the full training pipeline, model-selection rationale, and final model card.

## Reproducibility Notes

Machine-specific paths from the original training server were sanitized for public release.

See `metadata/PATH_SANITIZATION.md`.

Raw training datasets, generated labels, training weights, and model binaries are intentionally excluded from Git history.

## Environment

Reference environment records are stored under `metadata/environment/`.
