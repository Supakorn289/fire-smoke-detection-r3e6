# Training / Export Reference Environment

- Python: 3.13.5
- Ultralytics: 8.4.95
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- ONNX: 1.22.0
- ONNX Runtime: 1.28.0
- OpenVINO: 2026.2.1-21919-ede283a88e3-releases/2026/2
- OpenCV: 5.0.0

## Notes

The production reference model is the frozen PyTorch `.pt` model.

ONNX export passed PT↔ONNX equivalence validation on the validation set.

OpenVINO was investigated for CPU deployment, but it was not approved as the frozen production backend at the time of this repository snapshot.
