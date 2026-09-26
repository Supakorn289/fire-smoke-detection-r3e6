# Engineering Decisions

## 1. Data quality before model optimization

FASDD curation addressed exact/visual/near duplicates, conflicts, split leakage, and dataset validation before the final model lineage was established.

## 2. R1 was an architecture study

R1 compared YOLOv8n, YOLOv8s, and YOLOv8m.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.714 | 0.643 | 0.701 | 0.424 |
| YOLOv8s | 0.717 | 0.645 | 0.708 | 0.428 |
| YOLOv8m | 0.738 | 0.656 | 0.724 | 0.447 |

YOLOv8s became the downstream lineage. The choice was not based only on maximum validation mAP; deployment practicality and later refinement were also part of the workflow.

## 3. Hard-negative mining was manually verified

False-alert candidates were mined from previously unseen negative images and manually reviewed before they were used as verified hard negatives.

## 4. R2.1 protected positive sensitivity

R2 focused on false-positive reduction. R2.1 was then used as a preserve-positive refinement so false-positive gains would not come only from making the detector overly conservative.

## 5. External negative-only data was used for robustness

SkyFinder supplied external outdoor negative scenes. Camera-level separation prevented overlap between proxy training cameras and held-out proxy cameras.

## 6. R3-E6 used multi-objective selection

At confidence `0.25`:

| Model | Held-out proxy image FPR |
|---|---:|
| R2.1 baseline | 0.2685 |
| R3-E6 | 0.1502 |

R3-E6 reduced proxy image FPR by approximately 44% relative to R2.1 while keeping validation performance close to the preceding lineage.

## 7. Model freeze happened before Final TEST

The final checkpoint, operating point, class contract, and model hash were frozen before the Final TEST was opened.

Final model SHA256:

`49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee354d9d7a5ae1cb8c53a22183`

## 8. Deployment validation stayed off TEST

PT↔ONNX equivalence was evaluated on validation data, not on the closed Final TEST.

Recorded agreement:

- PT detections: `4,936`
- ONNX detections: `4,936`
- matched detections: `4,936`
- match rate: `1.0`
- mean bbox IoU: approximately `0.999999`

## 9. Binary models stay outside Git history

The `.pt` and `.onnx` files are distributed through GitHub Releases and verified with SHA256.

## 10. The research snapshot is DOI-backed

Zenodo DOI:

`10.5281/zenodo.22954656`
