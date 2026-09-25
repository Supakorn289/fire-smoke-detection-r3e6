# Reproducibility and Research Governance

## Final TEST governance

The Final TEST was locked before final evaluation.

It is not used for training, checkpoint selection, hyperparameter tuning, confidence tuning, ONNX equivalence testing, or deployment backend optimization.

## Model identity

PyTorch SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

ONNX SHA256:

`e887a759bcc65c89803c8ea0aa3d8891bcb0e60a5fd334433b2d6d912ba3bc79`

Verify downloaded release assets against `model/SHA256SUMS.txt`.

## Filesystem sanitization

Public copies replace machine-specific filesystem paths with portable placeholders or repository-relative paths.

Path sanitization does not change metrics, hashes, model-selection records, or Final TEST results.

## Metric protocol warning

Ultralytics standard Precision/Recall/mAP and fixed-confidence GT recall are different protocols.

Do not compare or combine them without stating the protocol.

## Negative-image statistics

At confidence 0.25 on the Final TEST:

- negative images: `6,525`
- images with at least one alert: `1,592`
- image-level FPR: `0.2440`
- prediction boxes on negatives: `2,665`

The `2,665` prediction boxes are not an authoritative object-level confusion-matrix FP count.
