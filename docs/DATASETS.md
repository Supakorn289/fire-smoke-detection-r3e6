# Datasets and External Data Sources

This document separates historical exploratory data, the final training lineage, and auxiliary negative-only sources.

| Dataset / source | Project role | Raw data redistributed here? |
|---|---|---|
| D-Fire | Early bootstrap, inspection, R0 experiments | No |
| FASDD_CV | Main dataset lineage from curation through Final TEST | No |
| SkyFinder | Internet proxy-negative mining and held-out proxy evaluation | No |

## D-Fire

Upstream: https://github.com/gaia-solutions-on-demand/DFireDataset

D-Fire is an image dataset for fire and smoke detection with more than 21,000 images and YOLO-format annotations.

### Role in this project

D-Fire belongs to the early historical stage:

- dataset inspection;
- manual review / visual QA;
- bootstrap dataset construction;
- initial R0 teacher experiments;
- auto-curator calibration experiments.

Historical scripts include `inspect_dfire.py`, `review_dfire.py`, `select_dfire.py`, `visual_qa_dfire.py`, `build_bootstrap_v1.py`, `sanitize_bootstrap_boxes.py`, and `train_teachers_r0.py`.

A historical `dfire_2000_manifest.csv` exists for the R0 auto-curator workflow.

D-Fire is not the final locked TEST source.

### Citation

See the upstream D-Fire repository for current download, license, and citation instructions.

## FASDD / FASDD_CV

Upstream repository: https://github.com/openrsgis/FASDD
Paper DOI: https://doi.org/10.1080/10095020.2024.2347922

This project used the FASDD_CV lineage for the main detector-development pipeline.

### FASDD CLEAN V1

| Split | Images |
|---|---:|
| Train | 45,658 |
| Validation | 31,254 |
| Final TEST | 15,863 |
| **Total** | **92,775** |

Final TEST composition:

- Fire only: `2,090`
- Smoke only: `3,896`
- Fire + Smoke: `3,352`
- Negative: `6,525`
- Fire GT boxes: `9,005`
- Smoke GT boxes: `8,208`

FASDD_CV / FASDD CLEAN V1 is the core data lineage for R1, hard-negative mining, R2, R2.1, R3, and the locked Final TEST.

### Citation

Ming Wang, Peng Yue, Liangcun Jiang, Dayu Yu, Tianyu Tuo, Jian Li.
*An open flame and smoke detection dataset for deep learning in remote sensing based fire detection.*
Geo-spatial Information Science, 28(2), 511–526, 2025.
DOI: `10.1080/10095020.2024.2347922`

## SkyFinder

Zenodo: https://zenodo.org/records/5884485
Dataset DOI: https://doi.org/10.5281/zenodo.5884485

SkyFinder was used only as an internet proxy-negative source.

Project-specific proxy workflow:

- `2,000` images;
- `10` cameras;
- R2.1 mining at confidence `0.25`;
- `427` candidates manually reviewed;
- `318` verified proxy hard negatives added to R3 training;
- held-out proxy evaluation used `406` images from held-out camera IDs.

SkyFinder does not provide positive Fire/Smoke labels for this project.

### Citation

Radu Paul Mihail, Scott Workman, Zach Bessinger, Nathan Jacobs.
*Sky Segmentation in the Wild: An Empirical Study.*
IEEE WACV, 2016.
DOI: `10.1109/WACV.2016.7477637`

## Redistribution policy

Raw third-party dataset images and labels are intentionally not redistributed here. Obtain them from the upstream providers and follow their terms and citation requirements.
