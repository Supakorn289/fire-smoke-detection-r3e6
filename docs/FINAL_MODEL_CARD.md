# Final Model Card

## Model

Fire and smoke object detector based on YOLOv8s.

- Classes: `0=Fire`, `1=Smoke`
- Input image size: `768 × 768`
- Confidence threshold: `0.25`
- NMS IoU: `0.70`
- `max_det`: `300`
- Batch size: `1`

## Frozen PyTorch Artifact

File: `fire_smoke_r3_e6_final.pt`

SHA256: `49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

## Final Test Dataset

Total images: `15,863`

Composition:

- fire-only: `2,090`
- smoke-only: `3,896`
- fire + smoke: `3,352`
- negative: `6,525`

Ground-truth boxes:

- Fire: `9,005`
- Smoke: `8,208`

## Final Test Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.8109 | 0.7364 | 0.8044 | 0.5256 |
| Fire | 0.8068 | 0.7815 | 0.8473 | 0.5382 |
| Smoke | 0.8150 | 0.6914 | 0.7615 | 0.5130 |

Derived F1 values from the reported precision and recall:

- Overall: approximately `0.7719`
- Fire: approximately `0.7939`
- Smoke: approximately `0.7481`

These F1 values are derived values and are not separate canonical metrics from the original evaluator.

## Fixed Operating Point

At confidence 0.25 and IoU 0.5:

Fire:
- TP: `7,822`
- FN: `1,183`
- GT recall: `0.8686`

Smoke:
- TP: `6,361`
- FN: `1,847`
- GT recall: `0.7750`

These fixed-threshold recall values use a different protocol from the Ultralytics standard recall and must not be mixed directly.

## Negative-Image Behavior

- Negative test images: `6,525`
- Images with at least one alert at confidence 0.25: `1,592`
- Image-level false-positive rate: `0.2440`
- Prediction boxes on negative images: `2,665`

The 2,665 prediction boxes are not an authoritative object-level confusion-matrix false-positive count.

## Size-Aware Recall at Confidence 0.25

Fire:
- Small: `0.6451`
- Medium: `0.8925`
- Large: `0.9671`

Smoke:
- Small: `0.4176`
- Medium: `0.8344`
- Large: `0.7360`

Small smoke remains the most difficult size category.

## ONNX Deployment Artifact

File: `fire_smoke_r3_e6_final_768_fp32.onnx`

SHA256: `e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

The ONNX export passed equivalence validation against the frozen PyTorch model using validation data.

The final test set was not used for deployment backend optimization.

## Limitations

- Small smoke has substantially lower recall than medium smoke.
- Single-frame negative-image false alerts remain non-zero.
- Runtime performance depends strongly on deployment CPU and inference backend.
- Temporal confirmation can be applied at the system level, but its alert-level performance must be evaluated separately from the closed final-test image-level metrics.
