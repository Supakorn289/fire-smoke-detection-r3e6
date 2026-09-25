# Project History — 0 → 100%

## 0–10% — Problem framing and dataset exploration

The project began as a two-class object-detection task: `Fire` and `Smoke`.

Early work focused on dataset structure, annotation quality, class mapping, negative scenes, and YOLO-format validation.

## 10–20% — D-Fire bootstrap and R0

A bootstrap dataset was prepared from early D-Fire work.

Historical work included:

- D-Fire inspection / review / selection;
- bootstrap dataset construction and box sanitization;
- YOLOv8n / YOLOv8s / YOLOv8m R0 teacher training;
- auto-curator calibration experiments.

R0 is preserved as historical experimentation, not the final model-selection stage.

## 20–40% — FASDD adoption and cleaning

The project moved to FASDD_CV as its main dataset lineage.

The curation pipeline addressed exact duplicates, decoded-image equality, visual duplicates, near-duplicate candidates, conflict adjudication, split leakage, and final dataset validation.

This produced **FASDD CLEAN V1**.

## 40–50% — R1 architecture comparison

R1 compared YOLOv8n, YOLOv8s, and YOLOv8m under the same dataset/validation lineage.

YOLOv8s became the downstream lineage.

## 50–60% — Hard-negative mining

Previously unseen clean training negatives were evaluated and candidate false-positive scenes were manually reviewed.

Verified hard negatives were separated into training and held-out validation subsets.

## 60–70% — R2

R2 initialized from R1 YOLOv8s and incorporated verified hard negatives to reduce false positives.

## 70–75% — R2.1

R2.1 initialized from R2 and focused on preserving positive sensitivity while keeping useful hard-negative learning.

## 75–82% — SkyFinder proxy-negative challenge

A 2,000-image / 10-camera SkyFinder proxy pool was created.

The R2.1 model mined candidate alerts, and 427 images were manually reviewed.

Camera-level separation prevented overlap between proxy training and proxy holdout cameras.

## 82–90% — R3

R3 initialized from R2.1 and incorporated 318 verified proxy hard negatives.

Multiple checkpoints were compared using main validation and held-out negative behavior.

`epoch6.pt` was selected as a multi-objective trade-off.

## 90–95% — Model freeze

R3 `epoch6.pt` was frozen as `fire_smoke_r3_e6_final.pt`.

SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

## 95–98% — Locked Final TEST

The frozen model was evaluated once on the locked 15,863-image Final TEST.

- Precision: `0.8109`
- Recall: `0.7364`
- mAP50: `0.8044`
- mAP50-95: `0.5256`

## 98–100% — Deployment export and release

The frozen PyTorch model was exported to FP32 ONNX, opset 17.

PT↔ONNX equivalence was checked using validation data.

The final artifacts were published as GitHub Release `v1.0.0-r3e6`:

- `fire_smoke_r3_e6_final.pt`
- `fire_smoke_r3_e6_final_768_fp32.onnx`
