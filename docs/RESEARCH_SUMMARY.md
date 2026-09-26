# Research Summary

## Scope

This repository is a research software and experimental-artifact repository for a two-class Fire/Smoke detector.

It preserves the development lineage from early dataset exploration and bootstrap experiments through dataset curation, controlled model comparison, hard-negative refinement, checkpoint selection, locked final testing, and deployment export.

## Research objective

The project investigates how to build a practical detector for:

- `Fire`
- `Smoke`

with attention to positive detection performance and false-alert behavior in negative outdoor scenes.

## Data sources

| Source | Role |
|---|---|
| D-Fire | Historical bootstrap / R0 experiments |
| FASDD_CV | Main dataset lineage from curation through Final TEST |
| SkyFinder | External negative-only proxy source for false-alert mining |

Detailed provenance: `docs/DATASETS.md`.

## Experimental stages

| Stage | Purpose | Outcome |
|---|---|---|
| Bootstrap / R0 | Early dataset and teacher experiments | Historical baseline |
| FASDD curation | Duplicate/conflict/leakage control | FASDD CLEAN V1 |
| R1 | YOLOv8n/s/m comparison | YOLOv8s downstream lineage |
| Hard-negative mining | Mine verified false-alert scenes | Hard-negative train/holdout |
| R2 | Reduce false positives | FP-focused refinement |
| R2.1 | Preserve positive sensitivity | Positive-preserving refinement |
| SkyFinder proxy | External negative challenge | Proxy hard negatives |
| R3 | Refine with proxy negatives | Multiple checkpoints evaluated |
| Final selection | Multi-objective selection | R3 `epoch6.pt` frozen |
| Final TEST | Locked one-time evaluation | Final metrics recorded |
| Deployment | Export + equivalence | PT↔ONNX PASS |

## Final model

- Architecture: `YOLOv8s`
- Classes: `0=Fire`, `1=Smoke`
- Input size: `768`
- Confidence: `0.25`
- NMS IoU: `0.70`
- `max_det`: `300`
- Frozen lineage: `R3 epoch6`

PyTorch SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

ONNX SHA256:

`e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

## Locked Final TEST

- Images: `15,863`
- Fire GT boxes: `9,005`
- Smoke GT boxes: `8,208`
- Negative images: `6,525`

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.8109 | 0.7364 | 0.8044 | 0.5256 |
| Fire | 0.8068 | 0.7815 | 0.8473 | 0.5382 |
| Smoke | 0.8150 | 0.6914 | 0.7615 | 0.5130 |

Derived F1:

- Overall: approximately `0.7719`
- Fire: approximately `0.7939`
- Smoke: approximately `0.7481`

## Negative-image behavior

At confidence `0.25`:

- negative images: `6,525`
- alert images: `1,592`
- image-level FPR: `0.2440`
- Fire image-level FPR: `0.1557`
- Smoke image-level FPR: `0.1054`
- prediction boxes on negative images: `2,665`

The `2,665` boxes are not presented as an authoritative object-level confusion-matrix FP count.

## R3 selection evidence

Held-out proxy image FPR at confidence `0.25`:

- R2.1 baseline: approximately `0.2685`
- selected R3-E6: approximately `0.1502`

Relative reduction: approximately `44%`.

Selection was a multi-objective trade-off, not simply maximum mAP.

## Deployment validation

The frozen model was exported to FP32 ONNX, opset `17`.

PT↔ONNX equivalence used validation data, not Final TEST data.

Recorded agreement:

- PT detections: `4,936`
- ONNX detections: `4,936`
- matched detections: `4,936`
- match rate: `1.0`
- mean bbox IoU: approximately `0.999999`
- maximum standard-metric delta: approximately `4.24e-6`

## Research governance

The Final TEST was not reused for:

- training;
- checkpoint selection;
- hyperparameter tuning;
- confidence tuning;
- ONNX equivalence testing;
- deployment backend optimization.

Evidence:

- `docs/REPRODUCIBILITY.md`
- `metadata/datasets/final_test_TEST_LOCK.json`
- `results/final_test/model_freeze/FINAL_MODEL_FREEZE.json`

## Artifact map

```text
configs/                  dataset/training configurations
scripts/bootstrap_r0/     D-Fire / bootstrap / R0
scripts/dataset_curation/ FASDD cleaning and leakage control
scripts/r1/               architecture comparison
scripts/hard_negative/    hard-negative mining
scripts/r2/               R2 refinement
scripts/r21/              R2.1 refinement
scripts/r3/               proxy-negative / R3
scripts/final_evaluation/ final evaluation
scripts/deployment/       ONNX / equivalence / benchmark

results/bootstrap_r0/     early artifacts
results/r1/               R1 results
results/r2/               R2 results
results/r21/              R2.1 results
results/r3/               R3 results
results/final_test/       locked test records
results/deployment/       deployment records
```

## Model distribution

GitHub Release:

https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.0-r3e6

Assets:

- `fire_smoke_r3_e6_final.pt`
- `fire_smoke_r3_e6_final_768_fp32.onnx`

## Limitations

- Small smoke remains the most difficult size category.
- Single-frame false alerts remain non-zero.
- Runtime depends on deployment hardware/backend.
- Fixed-confidence recall and standard Ultralytics recall are different protocols.
- Temporal confirmation belongs to the system layer and requires separate evaluation.

## Citation

Canonical research archive:

> Supakorn Prammano. *Fire & Smoke Detection R3-E6*. Version 1.0.1. Zenodo, 2026. DOI: `10.5281/zenodo.22954656`

DOI:

https://doi.org/10.5281/zenodo.22954656

Citation metadata is also available in the repository `CITATION.cff`.
