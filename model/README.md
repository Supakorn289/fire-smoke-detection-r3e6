# Final Model Artifacts

## Download

GitHub Release: [v1.0.0-r3e6](https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/tag/v1.0.0-r3e6)

### PyTorch

`fire_smoke_r3_e6_final.pt`

SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

### ONNX FP32

`fire_smoke_r3_e6_final_768_fp32.onnx`

SHA256:

`e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

ONNX opset: `17`

## Model contract

- Architecture: `YOLOv8s`
- Classes: `0=Fire`, `1=Smoke`
- Image size: `768`
- Confidence threshold: `0.25`
- NMS IoU: `0.70`
- `max_det`: `300`
- Batch: `1`

The ONNX export was verified against the frozen PyTorch model using validation data. Final TEST data was not used for deployment equivalence testing.

## Integrity check

```bash
sha256sum fire_smoke_r3_e6_final.pt
sha256sum fire_smoke_r3_e6_final_768_fp32.onnx
```

Compare the output with `SHA256SUMS.txt`.
