<div align="center">

# 🔥 Fire & Smoke Detection — R3-E6

### End-to-end computer vision research: dataset engineering → model refinement → locked evaluation → deployable release

[![DOI](https://zenodo.org/badge/1384914946.svg)](https://doi.org/10.5281/zenodo.22954656)
[![Model](https://img.shields.io/badge/model-YOLOv8s-111827)](./docs/FINAL_MODEL_CARD.md)
[![Classes](https://img.shields.io/badge/classes-Fire%20%7C%20Smoke-dc2626)](./docs/FINAL_MODEL_CARD.md)
[![Input](https://img.shields.io/badge/input-768%C3%97768-2563eb)](./docs/FINAL_MODEL_CARD.md)
[![Final TEST mAP50](https://img.shields.io/badge/Final%20TEST%20mAP50-0.8044-16a34a)](./results/final_test/FINAL_TEST_SUMMARY.txt)
[![Model Release](https://img.shields.io/badge/model%20release-v1.0.0--r3e6-7c3aed)](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.0-r3e6)
[![Research Archive](https://img.shields.io/badge/research%20archive-v1.0.1-0f766e)](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.1)

**Python · PyTorch · Ultralytics YOLOv8 · ONNX · ONNX Runtime · OpenCV**

</div>

---

## At a glance

| | |
|---|---|
| **Task** | Two-class object detection: `Fire` and `Smoke` |
| **Final architecture** | YOLOv8s |
| **Main data lineage** | FASDD_CV → FASDD CLEAN V1 |
| **Final TEST** | 15,863 locked images |
| **Final mAP50** | **0.8044** |
| **Final mAP50-95** | **0.5256** |
| **External negative robustness** | SkyFinder proxy hard-negative refinement |
| **Deployment** | PyTorch + validated FP32 ONNX |
| **Research archive DOI** | [`10.5281/zenodo.22954656`](https://doi.org/10.5281/zenodo.22954656) |

This repository is both a **portfolio project** and a **research artifact**. It preserves the full engineering path from early D-Fire/bootstrap experiments to a frozen model, locked final evaluation, ONNX validation, and DOI-backed research archive.

## What this project demonstrates

| Area | Evidence |
|---|---|
| **Dataset engineering** | duplicate analysis, conflict adjudication, leakage control, sanitization |
| **Model experimentation** | R0/R1 YOLOv8n-s-m comparisons, R2/R2.1/R3 refinement |
| **False-positive reduction** | verified hard-negative mining + external proxy-negative mining |
| **Evaluation discipline** | pre-TEST freeze, locked Final TEST, separate validation diagnostics |
| **Deployment engineering** | PyTorch → ONNX export, PT↔ONNX equivalence, CPU benchmark |
| **Research reproducibility** | configs, scripts, artifacts, hashes, environments, DOI |
| **Model distribution** | downloadable `.pt` and `.onnx` assets with SHA256 |

# Download the final model

Model release: [`v1.0.0-r3e6`](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.0-r3e6)

| Artifact | Format | Approx. size | SHA256 |
|---|---|---:|---|
| `fire_smoke_r3_e6_final.pt` | PyTorch / Ultralytics | 85.35 MiB | `49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183` |
| `fire_smoke_r3_e6_final_768_fp32.onnx` | ONNX FP32, opset 17 | 42.71 MiB | `e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79` |

Reference operating point:

```text
Architecture : YOLOv8s
Classes      : 0 = Fire, 1 = Smoke
Image size   : 768
Confidence   : 0.25
NMS IoU      : 0.70
max_det      : 300
Batch        : 1
```

# Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

Download the frozen model:

```bash
curl -L -o fire_smoke_r3_e6_final.pt \
  https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/download/v1.0.0-r3e6/fire_smoke_r3_e6_final.pt
```

Run inference:

```bash
python examples/infer_image.py \
  --model fire_smoke_r3_e6_final.pt \
  --source path/to/image.jpg
```

# Final results

## Locked Final TEST

| Metric | All | Fire | Smoke |
|---|---:|---:|---:|
| Precision | **0.8109** | 0.8068 | 0.8150 |
| Recall | **0.7364** | 0.7815 | 0.6914 |
| mAP50 | **0.8044** | 0.8473 | 0.7615 |
| mAP50-95 | **0.5256** | 0.5382 | 0.5130 |

Final TEST composition:

- `15,863` images
- `9,005` Fire GT boxes
- `8,208` Smoke GT boxes
- `6,525` negative images

Derived overall F1: approximately **0.7719**.

> The Final TEST was opened only after checkpoint selection and model freeze. It was not reused for training, tuning, checkpoint selection, ONNX equivalence testing, or deployment optimization.

## False-alert refinement

At confidence `0.25` on held-out SkyFinder proxy negatives:

| Model | Proxy image FPR |
|---|---:|
| R2.1 baseline | 0.2685 |
| **R3-E6 selected** | **0.1502** |

Relative reduction: approximately **44%**.

# Development history — 0 → 100%

```mermaid
flowchart LR
    A["D-Fire / Bootstrap"] --> B["R0"]
    B --> C["FASDD audit"]
    C --> D["FASDD CLEAN V1"]
    D --> E["R1 n / s / m"]
    E --> F["Hard negatives"]
    F --> G["R2"]
    G --> H["R2.1"]
    H --> I["SkyFinder proxy"]
    I --> J["R3"]
    J --> K["E6 freeze"]
    K --> L["Locked TEST"]
    L --> M["ONNX"]
    M --> N["Release + DOI"]
```

See:

- [`docs/PROJECT_HISTORY.md`](./docs/PROJECT_HISTORY.md)
- [`docs/TRAINING_PIPELINE.md`](./docs/TRAINING_PIPELINE.md)
- [`docs/ENGINEERING_DECISIONS.md`](./docs/ENGINEERING_DECISIONS.md)

# Datasets and provenance

| Source | Role | Final lineage? |
|---|---|---|
| **D-Fire** | Bootstrap / R0 historical experimentation | Historical / pre-R1 |
| **FASDD_CV** | Main Fire/Smoke dataset lineage | **Yes** |
| **SkyFinder** | External negative-only proxy pool | Auxiliary negative source |

FASDD CLEAN V1:

| Split | Images |
|---|---:|
| Train | 45,658 |
| Validation | 31,254 |
| Final TEST | 15,863 |
| **Total** | **92,775** |

Raw third-party datasets are not redistributed.

See [`docs/DATASETS.md`](./docs/DATASETS.md) and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

# Visual experiment artifacts

<p align="center">
  <img src="results/r3/teacher_s_r3_proxy_v1/results.png" width="48%" alt="R3 training curves">
  <img src="results/r3/teacher_s_r3_proxy_v1/confusion_matrix_normalized.png" width="48%" alt="R3 normalized confusion matrix">
</p>

R0 and R1 architecture artifacts are under:

- [`results/bootstrap_r0/`](./results/bootstrap_r0/)
- [`results/r1/`](./results/r1/)

# Deployment validation

The frozen PyTorch model was exported to FP32 ONNX, opset `17`.

| Check | Result |
|---|---:|
| PT detections | 4,936 |
| ONNX detections | 4,936 |
| Matched detections | 4,936 |
| Match rate | 1.0 |
| Mean bbox IoU | ≈ 0.999999 |
| Max standard-metric delta | ≈ 4.24e-6 |

Deployment artifacts: [`results/deployment/`](./results/deployment/)

# Research reproducibility

The public snapshot preserves scripts, configs, R0→R3 result artifacts, freeze/test-lock records, model hashes, environment snapshots, PT↔ONNX evidence, and CPU benchmark results.

See:

- [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md)
- [`metadata/PATH_SANITIZATION.md`](./metadata/PATH_SANITIZATION.md)
- [`model/SHA256SUMS.txt`](./model/SHA256SUMS.txt)

# Research archive and citation

**DOI:** [`10.5281/zenodo.22954656`](https://doi.org/10.5281/zenodo.22954656)

> Supakorn Prammano. *Fire & Smoke Detection R3-E6*. Version 1.0.1. Zenodo, 2026. DOI: [10.5281/zenodo.22954656](https://doi.org/10.5281/zenodo.22954656)

Research overview: [`docs/RESEARCH_SUMMARY.md`](./docs/RESEARCH_SUMMARY.md)

# Releases

- [`v1.0.0-r3e6`](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.0-r3e6) — frozen PyTorch + ONNX model binaries
- [`v1.0.1`](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.1) — DOI-backed research archive

# Limitations

- Small smoke remains the most difficult size category.
- Single-frame negative-image false alerts remain non-zero.
- Fixed-confidence GT recall and Ultralytics standard recall are different protocols.
- The `2,665` prediction boxes on negative Final TEST images are not an authoritative object-level FP confusion-matrix count.
- Runtime depends on hardware/backend.
- Temporal confirmation belongs to the system layer and requires separate evaluation.

## Documentation index

- [Research summary](./docs/RESEARCH_SUMMARY.md)
- [Project history](./docs/PROJECT_HISTORY.md)
- [Datasets and citations](./docs/DATASETS.md)
- [Engineering decisions](./docs/ENGINEERING_DECISIONS.md)
- [Training pipeline](./docs/TRAINING_PIPELINE.md)
- [Model selection](./docs/MODEL_SELECTION.md)
- [Final model card](./docs/FINAL_MODEL_CARD.md)
- [Reproducibility](./docs/REPRODUCIBILITY.md)
- [Third-party notices](./THIRD_PARTY_NOTICES.md)

> **License note:** this repository does not currently declare a repository-wide source-code license. Third-party datasets and dependencies remain subject to their upstream terms.
