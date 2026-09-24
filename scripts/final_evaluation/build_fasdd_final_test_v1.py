#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "reports/fasdd/clean_v1/test.txt"
)

SOURCE = (
    ROOT
    / "sources/fasdd/FASDD_CV"
)

SOURCE_IMAGES = (
    SOURCE
    / "images"
)

SOURCE_LABELS = (
    SOURCE
    / "annotations/YOLO_CV/labels"
)

OUT = (
    ROOT
    / "datasets/fasdd_final_test_locked_v1"
)

REPORT = (
    ROOT
    / "reports/final_test_v1"
)


EXPECTED = {
    "images": 15863,

    "fire_only": 2090,
    "smoke_only": 3896,
    "fire_smoke": 3352,
    "negative": 6525,

    "fire_boxes": 9005,
    "smoke_boxes": 8208,
}


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def resolve_image(stem):

    found = []

    for ext in IMAGE_EXTENSIONS:

        p = (
            SOURCE_IMAGES
            / f"{stem}{ext}"
        )

        if p.exists():
            found.append(
                p.resolve()
            )


    if len(found) == 0:

        raise FileNotFoundError(
            f"Missing source image: {stem}"
        )


    if len(found) > 1:

        hashes = {
            sha256(p)
            for p in found
        }

        if len(hashes) != 1:

            raise RuntimeError(
                "Multiple conflicting source "
                f"images for stem: {stem}\n"
                + "\n".join(
                    str(p)
                    for p in found
                )
            )


    return found[0]


def parse_label(path):

    text = path.read_text(
        encoding="utf-8"
    ).strip()


    if not text:

        return []


    classes = []


    for line_no, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        parts = line.split()


        if len(parts) != 5:

            raise RuntimeError(
                f"Invalid YOLO row: {path}\n"
                f"line={line_no}\n"
                f"{line}"
            )


        cls = int(
            float(parts[0])
        )


        if cls not in (0, 1):

            raise RuntimeError(
                f"Unexpected class={cls} "
                f"in {path}"
            )


        # Validate numeric coordinates.
        values = [
            float(x)
            for x in parts[1:]
        ]


        for value in values:

            if not (
                0.0 <= value <= 1.0
            ):

                raise RuntimeError(
                    "Out-of-range TEST YOLO "
                    f"value={value} in {path}"
                )


        classes.append(
            cls
        )


    return classes


parser = argparse.ArgumentParser()

parser.add_argument(
    "--rebuild",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# PRE-FLIGHT
# ============================================================

print("=" * 100)
print(
    "BUILD FASDD CLEAN V1 "
    "FINAL LOCKED TEST"
)
print("=" * 100)

print(
    "Manifest:",
    MANIFEST
)

print(
    "Source  :",
    SOURCE
)

print(
    "Images  :",
    SOURCE_IMAGES
)

print(
    "Labels  :",
    SOURCE_LABELS
)

print()


for path in [
    MANIFEST,
    SOURCE,
    SOURCE_IMAGES,
    SOURCE_LABELS,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


print(
    "Resolved source:",
    SOURCE.resolve()
)


# ============================================================
# READ FROZEN CLEAN-V1 TEST MANIFEST
# ============================================================

stems = [
    line.strip()
    for line in MANIFEST.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


if len(stems) != EXPECTED["images"]:

    raise RuntimeError(
        "TEST manifest count mismatch: "
        f"{len(stems)} != "
        f"{EXPECTED['images']}"
    )


if len(set(stems)) != len(stems):

    raise RuntimeError(
        "Duplicate stem detected in "
        "Clean V1 TEST manifest"
    )


manifest_sha256 = sha256(
    MANIFEST
)


print()
print(
    "TEST stems:",
    len(stems)
)

print(
    "Manifest SHA256:",
    manifest_sha256
)


# ============================================================
# RESOLVE EXACT SOURCE FILES
# ============================================================

records = []

stats = {
    "fire_only": 0,
    "smoke_only": 0,
    "fire_smoke": 0,
    "negative": 0,
    "fire_boxes": 0,
    "smoke_boxes": 0,
}


print()
print("=" * 100)
print(
    "VERIFYING ALL TEST SOURCE FILES"
)
print("=" * 100)


for index, stem in enumerate(
    stems,
    start=1,
):

    image = resolve_image(
        stem
    )


    label = (
        SOURCE_LABELS
        / f"{stem}.txt"
    )


    if not label.exists():

        raise FileNotFoundError(
            f"Missing source label: {label}"
        )


    label = label.resolve()


    classes = parse_label(
        label
    )


    fire_count = classes.count(
        0
    )

    smoke_count = classes.count(
        1
    )


    stats[
        "fire_boxes"
    ] += fire_count

    stats[
        "smoke_boxes"
    ] += smoke_count


    has_fire = (
        fire_count > 0
    )

    has_smoke = (
        smoke_count > 0
    )


    if (
        has_fire
        and has_smoke
    ):

        category = (
            "fire_smoke"
        )


    elif has_fire:

        category = (
            "fire_only"
        )


    elif has_smoke:

        category = (
            "smoke_only"
        )


    else:

        category = (
            "negative"
        )


    stats[
        category
    ] += 1


    records.append({
        "stem":
            stem,

        "image":
            str(image),

        "label":
            str(label),

        "category":
            category,

        "fire_boxes":
            fire_count,

        "smoke_boxes":
            smoke_count,
    })


    if (
        index % 2000 == 0
        or index == len(stems)
    ):

        print(
            f"\rVerified "
            f"{index}/{len(stems)}",
            end="",
            flush=True,
        )


print("\n")


# ============================================================
# FROZEN IDENTITY ASSERTIONS
# ============================================================

actual = {
    "images":
        len(records),

    **stats,
}


print("=" * 100)
print(
    "FROZEN TEST IDENTITY"
)
print("=" * 100)


for key in [
    "images",
    "fire_only",
    "smoke_only",
    "fire_smoke",
    "negative",
    "fire_boxes",
    "smoke_boxes",
]:

    expected = EXPECTED[
        key
    ]

    value = actual[
        key
    ]

    status = (
        "PASS"
        if value == expected
        else "FAIL"
    )

    print(
        f"{key:<14}: "
        f"{value:>6} "
        f"(expected {expected:>6}) "
        f"{status}"
    )


if actual != EXPECTED:

    raise RuntimeError(
        "\nFASDD CLEAN V1 TEST "
        "IDENTITY MISMATCH.\n"
        "DO NOT RUN FINAL TEST."
    )


# ============================================================
# BUILD LOCKED DATASET
# ============================================================

if OUT.exists():

    if not args.rebuild:

        raise RuntimeError(
            f"{OUT} already exists.\n"
            "Use --rebuild only if rebuilding "
            "before FINAL TEST execution."
        )

    shutil.rmtree(
        OUT
    )


IMAGE_OUT = (
    OUT
    / "images/test"
)

LABEL_OUT = (
    OUT
    / "labels/test"
)


IMAGE_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

LABEL_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "BUILDING READ-ONLY STYLE "
    "LOCKED TEST DATASET"
)
print("=" * 100)


for index, record in enumerate(
    records,
    start=1,
):

    source_image = Path(
        record["image"]
    )

    source_label = Path(
        record["label"]
    )


    image_dest = (
        IMAGE_OUT
        / (
            record["stem"]
            + source_image.suffix.lower()
        )
    )


    label_dest = (
        LABEL_OUT
        / f"{record['stem']}.txt"
    )


    os.symlink(
        source_image,
        image_dest,
    )

    os.symlink(
        source_label,
        label_dest,
    )


    if (
        index % 2000 == 0
        or index == len(records)
    ):

        print(
            f"\rLinked "
            f"{index}/{len(records)}",
            end="",
            flush=True,
        )


print("\n")


# ============================================================
# DATA YAML
# ============================================================

DATA_YAML = (
    OUT
    / "data.yaml"
)


DATA_YAML.write_text(
    "\n".join([
        f"path: {OUT}",
        "",
        "# FINAL FASDD CLEAN V1 TEST",
        "# DO NOT USE FOR TRAINING",
        "# DO NOT USE FOR VALIDATION TUNING",
        "",
        "train: images/test",
        "val: images/test",
        "test: images/test",
        "",
        "nc: 2",
        "names:",
        "  0: fire",
        "  1: smoke",
        "",
    ]),
    encoding="utf-8",
)


# ============================================================
# MACHINE-READABLE LOCK RECORD
# ============================================================

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)


LOCK_RECORD = {
    "status":
        "FASDD_FINAL_TEST_DATASET_LOCKED",

    "dataset":
        "FASDD Clean V1",

    "source":
        str(
            SOURCE.resolve()
        ),

    "manifest":
        str(MANIFEST),

    "manifest_sha256":
        manifest_sha256,

    "class_contract": {
        "0": "fire",
        "1": "smoke",
    },

    "counts":
        actual,

    "test_dataset":
        str(OUT),

    "data_yaml":
        str(DATA_YAML),

    "policy": (
        "This dataset is the frozen final TEST set. "
        "It must never be used for training, "
        "checkpoint selection, hyperparameter tuning, "
        "or confidence-threshold selection."
    ),
}


LOCK_JSON = (
    OUT
    / "TEST_LOCK.json"
)


LOCK_JSON.write_text(
    json.dumps(
        LOCK_RECORD,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL SYMLINK COUNT CHECK
# ============================================================

built_images = sum(
    1
    for p in IMAGE_OUT.iterdir()
    if p.is_symlink()
)

built_labels = sum(
    1
    for p in LABEL_OUT.iterdir()
    if p.is_symlink()
)


if built_images != EXPECTED["images"]:

    raise RuntimeError(
        "Built image count mismatch"
    )


if built_labels != EXPECTED["images"]:

    raise RuntimeError(
        "Built label count mismatch"
    )


print("=" * 100)
print(
    "FASDD FINAL TEST DATASET SUMMARY"
)
print("=" * 100)

print(
    "Images      :",
    actual["images"]
)

print(
    "Fire only   :",
    actual["fire_only"]
)

print(
    "Smoke only  :",
    actual["smoke_only"]
)

print(
    "Fire+Smoke  :",
    actual["fire_smoke"]
)

print(
    "Negative    :",
    actual["negative"]
)

print(
    "Fire boxes  :",
    actual["fire_boxes"]
)

print(
    "Smoke boxes :",
    actual["smoke_boxes"]
)

print()

print(
    "Manifest SHA256:",
    manifest_sha256
)

print()

print(
    "Dataset:",
    OUT
)

print(
    "YAML   :",
    DATA_YAML
)

print(
    "Lock   :",
    LOCK_JSON
)

print()

print(
    "RESULT: FASDD FINAL TEST "
    "DATASET IDENTITY PASS"
)

print("=" * 100)
