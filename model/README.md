# Final Model Artifacts

## Frozen PyTorch model

File:

`fire_smoke_r3_e6_final.pt`

SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

Architecture:

`YOLOv8s`

Classes:

- 0: Fire
- 1: Smoke

Reference operating configuration:

- Image size: 768
- Confidence threshold: 0.25
- NMS IoU: 0.70
- max_det: 300
- Batch: 1

## ONNX export

File:

`fire_smoke_r3_e6_final_768_fp32.onnx`

SHA256:

`e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

The ONNX model was verified against the frozen PyTorch model on validation data.

Model binaries are intended to be distributed separately as GitHub Release assets rather than committed repeatedly into Git history.
