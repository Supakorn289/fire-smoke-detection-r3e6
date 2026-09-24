#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

PRED = (
    ROOT
    / "reports/internet_proxy_negative_v1/"
      "r21_mining/internet_proxy_r21_predictions.csv"
)

BASE = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1"
)

OUT = (
    ROOT
    / "datasets/fasdd_r3_proxy_v1"
)

REPORT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
HARD_THRESHOLD = 0.25
EXPECTED_TOTAL = 2000
EXPECTED_HARD = 427


def stable_camera_key(camera):
    return hashlib.sha256(
        f"{SEED}:{camera}".encode()
    ).hexdigest()


if not PRED.exists():
    raise FileNotFoundError(PRED)

if not BASE.exists():
    raise FileNotFoundError(BASE)


with PRED.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )


if len(rows) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL}, got {len(rows)}"
    )


cameras = sorted({
    row["camera_id"]
    for row in rows
})


if len(cameras) != 10:
    raise RuntimeError(
        f"Expected 10 cameras, got {len(cameras)}"
    )


# ============================================================
# CAMERA-LEVEL SPLIT
# deterministic, independent of model performance
# ============================================================

camera_order = sorted(
    cameras,
    key=stable_camera_key,
)

holdout_cameras = set(
    camera_order[:2]
)

train_cameras = set(
    camera_order[2:]
)


hard_rows = [
    row
    for row in rows
    if float(row["top_conf"])
    >= HARD_THRESHOLD
]


if len(hard_rows) != EXPECTED_HARD:
    raise RuntimeError(
        f"Expected {EXPECTED_HARD} hard rows, "
        f"got {len(hard_rows)}"
    )


train_hard = [
    row
    for row in hard_rows
    if row["camera_id"]
    in train_cameras
]

holdout_hard = [
    row
    for row in hard_rows
    if row["camera_id"]
    in holdout_cameras
]

holdout_all = [
    row
    for row in rows
    if row["camera_id"]
    in holdout_cameras
]


print("=" * 96)
print("R3 INTERNET PROXY CAMERA SPLIT")
print("=" * 96)

print(
    "TRAIN cameras  :",
    sorted(train_cameras)
)

print(
    "HOLDOUT cameras:",
    sorted(holdout_cameras)
)

print()

print(
    "Verified hard negatives total:",
    len(hard_rows)
)

print(
    "R3 train hard negatives      :",
    len(train_hard)
)

print(
    "Held-out hard alerts         :",
    len(holdout_hard)
)

print(
    "Held-out all proxy images    :",
    len(holdout_all)
)


# ============================================================
# SAVE SPLITS
# ============================================================

def save_csv(path, data):

    if not data:
        raise RuntimeError(
            f"No rows for {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(data[0].keys()),
        )

        writer.writeheader()
        writer.writerows(data)


save_csv(
    REPORT / "proxy_train_hard.csv",
    train_hard,
)

save_csv(
    REPORT / "proxy_holdout_hard.csv",
    holdout_hard,
)

save_csv(
    REPORT / "proxy_holdout_all.csv",
    holdout_all,
)


# ============================================================
# BUILD R3 DATASET
# ============================================================

if OUT.exists():
    shutil.rmtree(OUT)


shutil.copytree(
    BASE / "images",
    OUT / "images",
    symlinks=True,
)

shutil.copytree(
    BASE / "labels",
    OUT / "labels",
    symlinks=True,
)


# Delete old Ultralytics cache
for cache in OUT.rglob("*.cache"):
    cache.unlink()


proxy_image_dir = (
    OUT
    / "images/train/proxy_hard_v1"
)

proxy_label_dir = (
    OUT
    / "labels/train/proxy_hard_v1"
)

proxy_image_dir.mkdir(
    parents=True,
    exist_ok=True,
)

proxy_label_dir.mkdir(
    parents=True,
    exist_ok=True,
)


for i, row in enumerate(
    train_hard,
    start=1,
):

    source = Path(
        row["proxy_path"]
    )

    if not source.exists():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()

    new_stem = (
        f"proxy_cam{row['camera_id']}_"
        f"{i:04d}_"
        f"{row['sha256'][:10]}"
    )

    image_dest = (
        proxy_image_dir
        / f"{new_stem}{suffix}"
    )

    label_dest = (
        proxy_label_dir
        / f"{new_stem}.txt"
    )

    os.symlink(
        source.resolve(),
        image_dest,
    )

    # Human-verified negative:
    # valid empty YOLO label.
    label_dest.write_text(
        "",
        encoding="utf-8",
    )


# Remove cache again after modification
for cache in OUT.rglob("*.cache"):
    cache.unlink()


data_yaml = (
    OUT
    / "data.yaml"
)

data_yaml.write_text(
    "\n".join([
        f"path: {OUT}",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 2",
        "names:",
        "  0: fire",
        "  1: smoke",
        "",
    ]),
    encoding="utf-8",
)


image_exts = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp",
}


def count_images(path):

    return sum(
        1
        for p in path.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in image_exts
        )
    )


train_total = count_images(
    OUT / "images/train"
)

val_total = count_images(
    OUT / "images/val"
)


expected_train = (
    14544
    + len(train_hard)
)


if train_total != expected_train:
    raise RuntimeError(
        f"Train mismatch: "
        f"{train_total} != {expected_train}"
    )

if val_total != 3000:
    raise RuntimeError(
        f"VAL mismatch: {val_total}"
    )


print()
print("=" * 96)
print("R3 DATASET SUMMARY")
print("=" * 96)

print(
    "R2.1 base train      : 14544"
)

print(
    "Internet hardneg add :",
    len(train_hard)
)

print(
    "R3 train total       :",
    train_total
)

print(
    "Main FASDD VAL       :",
    val_total
)

print(
    "Proxy holdout images :",
    len(holdout_all)
)

print()

print(
    "Dataset:",
    OUT
)

print(
    "YAML   :",
    data_yaml
)

print()

print(
    "RESULT: R3 PROXY DATASET PASS"
)

print("=" * 96)
