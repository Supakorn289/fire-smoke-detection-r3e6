# Model Selection

## R1 Architecture Comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.714 | 0.643 | 0.701 | 0.424 |
| YOLOv8s | 0.717 | 0.645 | 0.708 | 0.428 |
| YOLOv8m | 0.738 | 0.656 | 0.724 | 0.447 |

YOLOv8s became the downstream model lineage after R1.

The decision was not based only on the largest validation mAP value; downstream deployment constraints and subsequent hard-negative refinement were also part of the development workflow.

## R2

R2 introduced verified hard negatives to reduce false-positive behavior.

## R2.1

R2.1 refined the R2 model with the objective of preserving positive sensitivity.

Validation summary:

- Precision: `0.693`
- Recall: `0.654`
- mAP50: `0.695`
- mAP50-95: `0.416`

## R3 Checkpoint Selection

R3 checkpoints were compared using both standard validation performance and held-out proxy-negative false-positive behavior.

R2.1 baseline proxy image FPR at confidence 0.25: approximately `0.2685`.

R3 checkpoint E6 proxy image FPR at confidence 0.25: approximately `0.1502`.

This corresponds to a relative reduction of approximately `44.0%` while retaining validation performance close to the preceding model.

Therefore R3 `epoch6.pt` was selected as the final checkpoint before the final test set was opened.

The final checkpoint selection was a multi-objective tradeoff and was not based only on maximum mAP.

## Final Model

Frozen artifact: `fire_smoke_r3_e6_final.pt`

SHA256: `49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`
