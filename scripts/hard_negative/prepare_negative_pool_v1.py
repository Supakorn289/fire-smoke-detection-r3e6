#!/usr/bin/env python3

from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# MANIFESTS
# ============================================================

CLEAN_TRAIN = (
    ROOT
    / "reports/fasdd/clean_v1/train.txt"
)

CLEAN_VAL = (
    ROOT
    / "reports/fasdd/clean_v1/val.txt"
)

CLEAN_TEST = (
    ROOT
    / "reports/fasdd/clean_v1/test.txt"
)

R1_TRAIN = (
    ROOT
    / "reports/fasdd/r1_selection_v1/"
    "fasdd_r1_train_12000.txt"
)


# ============================================================
# DATA ROOTS
# ============================================================

SOURCE_ROOT = (
    ROOT
    / "sources/fasdd/FASDD_CV"
)

CURATED_ROOT = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)


# ============================================================
# OUTPUT
# ============================================================

OUT = (
    ROOT
    / "reports/negative_v1/pool"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# MANIFEST READER
#
# Supports:
#
#   bothFireAndSmoke_CV000490
#
# or:
#
#   /some/path/bothFireAndSmoke_CV000490.jpg
#
# ============================================================

def read_manifest_stems(path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    stems = []

    seen = set()

    for raw in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        p = Path(line)

        # If it already has an extension/path,
        # Path.stem extracts the image ID.
        #
        # If it is already just a stem,
        # Path.stem returns the same string.
        stem = p.stem

        if not stem:
            raise RuntimeError(
                f"Invalid manifest entry: {line}"
            )

        if stem in seen:
            raise RuntimeError(
                f"Duplicate stem in {path}: "
                f"{stem}"
            )

        seen.add(stem)
        stems.append(stem)

    return stems


# ============================================================
# LOAD MANIFESTS
# ============================================================

print("=" * 92)
print("HARD NEGATIVE V1 — POOL PREPARATION")
print("=" * 92)

clean_train_stems = read_manifest_stems(
    CLEAN_TRAIN
)

clean_val_stems = read_manifest_stems(
    CLEAN_VAL
)

clean_test_stems = read_manifest_stems(
    CLEAN_TEST
)

r1_train_stems = read_manifest_stems(
    R1_TRAIN
)


print(
    "Clean train:",
    len(clean_train_stems)
)

print(
    "Clean val  :",
    len(clean_val_stems)
)

print(
    "Clean test :",
    len(clean_test_stems)
)

print(
    "R1 train   :",
    len(r1_train_stems)
)


# ============================================================
# EXPECTED FROZEN COUNTS
# ============================================================

if len(clean_train_stems) != 45658:
    raise RuntimeError(
        "Clean TRAIN count changed: "
        f"{len(clean_train_stems)} != 45658"
    )

if len(clean_val_stems) != 31254:
    raise RuntimeError(
        "Clean VAL count changed: "
        f"{len(clean_val_stems)} != 31254"
    )

if len(clean_test_stems) != 15863:
    raise RuntimeError(
        "Clean TEST count changed: "
        f"{len(clean_test_stems)} != 15863"
    )

if len(r1_train_stems) != 12000:
    raise RuntimeError(
        "R1 TRAIN count changed: "
        f"{len(r1_train_stems)} != 12000"
    )


# ============================================================
# SETS
# ============================================================

clean_train_set = set(
    clean_train_stems
)

clean_val_set = set(
    clean_val_stems
)

clean_test_set = set(
    clean_test_stems
)

r1_train_set = set(
    r1_train_stems
)


# ============================================================
# BASIC LEAKAGE CHECKS
# ============================================================

print()
print("Manifest overlap checks")

print(
    "TRAIN ∩ VAL :",
    len(
        clean_train_set
        & clean_val_set
    )
)

print(
    "TRAIN ∩ TEST:",
    len(
        clean_train_set
        & clean_test_set
    )
)

print(
    "VAL ∩ TEST  :",
    len(
        clean_val_set
        & clean_test_set
    )
)

if (
    clean_train_set
    & clean_val_set
):
    raise RuntimeError(
        "Clean TRAIN/VAL overlap"
    )

if (
    clean_train_set
    & clean_test_set
):
    raise RuntimeError(
        "Clean TRAIN/TEST overlap"
    )

if (
    clean_val_set
    & clean_test_set
):
    raise RuntimeError(
        "Clean VAL/TEST overlap"
    )


# ============================================================
# INDEX SOURCE IMAGES
#
# We only index images belonging to CLEAN TRAIN.
# ============================================================

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"FASDD source missing: "
        f"{SOURCE_ROOT}"
    )


print()
print("Indexing FASDD source images...")


image_index = {}

for path in SOURCE_ROOT.rglob("*"):

    if not path.is_file():
        continue

    if (
        path.suffix.lower()
        not in IMAGE_EXTS
    ):
        continue

    stem = path.stem

    if (
        stem
        not in clean_train_set
    ):
        continue

    if stem in image_index:

        old = image_index[
            stem
        ]

        # Same resolved file is harmless.
        if (
            old.resolve()
            != path.resolve()
        ):
            raise RuntimeError(
                "Duplicate source image stem:\n"
                f"{stem}\n"
                f"{old}\n"
                f"{path}"
            )

    image_index[
        stem
    ] = path.resolve()


print(
    "TRAIN images indexed:",
    len(image_index)
)


missing_images = (
    clean_train_set
    - set(image_index)
)

if missing_images:

    sample = sorted(
        missing_images
    )[:20]

    raise RuntimeError(
        "Missing source images: "
        f"{len(missing_images)}\n"
        f"Examples: {sample}"
    )


# ============================================================
# INDEX CURATED LABELS
#
# Curated labels are preferred because these are the
# sanitized/frozen R1-source annotations.
# ============================================================

if not CURATED_ROOT.exists():
    raise FileNotFoundError(
        f"Curated root missing: "
        f"{CURATED_ROOT}"
    )


print()
print("Indexing curated YOLO labels...")


label_index = {}

for path in CURATED_ROOT.rglob(
    "*.txt"
):

    if not path.is_file():
        continue

    stem = path.stem

    if (
        stem
        not in clean_train_set
    ):
        continue

    if stem in label_index:

        old = label_index[
            stem
        ]

        if (
            old.resolve()
            != path.resolve()
        ):
            raise RuntimeError(
                "Duplicate curated label stem:\n"
                f"{stem}\n"
                f"{old}\n"
                f"{path}"
            )

    label_index[
        stem
    ] = path.resolve()


print(
    "TRAIN labels indexed:",
    len(label_index)
)


missing_labels = (
    clean_train_set
    - set(label_index)
)

if missing_labels:

    sample = sorted(
        missing_labels
    )[:20]

    raise RuntimeError(
        "Missing curated labels: "
        f"{len(missing_labels)}\n"
        f"Examples: {sample}"
    )


# ============================================================
# IDENTIFY CLEAN NEGATIVES
#
# Negative = YOLO label exists but contains zero boxes.
# ============================================================

print()
print("Scanning CLEAN TRAIN negatives...")


clean_negatives = []

for stem in clean_train_stems:

    label_path = label_index[
        stem
    ]

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if text:
        continue

    clean_negatives.append(
        stem
    )


print(
    "Clean TRAIN negatives:",
    len(clean_negatives)
)


# ============================================================
# R1 NEGATIVE COUNT
# ============================================================

r1_negative_stems = [
    stem
    for stem in clean_negatives
    if stem in r1_train_set
]


print(
    "Negatives already in R1:",
    len(r1_negative_stems)
)


# ============================================================
# UNSEEN NEGATIVE POOL
# ============================================================

unseen_negative_stems = [
    stem
    for stem in clean_negatives
    if stem not in r1_train_set
]


print(
    "Unseen negatives:",
    len(unseen_negative_stems)
)


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

pool_set = set(
    unseen_negative_stems
)


r1_overlap = (
    pool_set
    & r1_train_set
)

val_overlap = (
    pool_set
    & clean_val_set
)

test_overlap = (
    pool_set
    & clean_test_set
)


print()
print("Negative-pool leakage checks")

print(
    "POOL ∩ R1 TRAIN:",
    len(r1_overlap)
)

print(
    "POOL ∩ VAL     :",
    len(val_overlap)
)

print(
    "POOL ∩ TEST    :",
    len(test_overlap)
)


if r1_overlap:
    raise RuntimeError(
        "Negative pool leaks into "
        "R1 TRAIN"
    )

if val_overlap:
    raise RuntimeError(
        "Negative pool leaks into "
        "VAL"
    )

if test_overlap:
    raise RuntimeError(
        "Negative pool leaks into "
        "TEST"
    )


# ============================================================
# EXPECTED NEGATIVE COUNTS
# ============================================================

# Frozen Clean V1 profile:
#
# CLEAN TRAIN negatives = 19,510
#
# R1 selected exactly 2,000 negative images.
#
# Therefore unseen clean-train negatives should be:
#
# 19,510 - 2,000 = 17,510
#

EXPECTED_CLEAN_NEG = 19510

EXPECTED_R1_NEG = 2000

EXPECTED_UNSEEN_NEG = 17510


if (
    len(clean_negatives)
    != EXPECTED_CLEAN_NEG
):

    raise RuntimeError(
        "Unexpected CLEAN TRAIN "
        "negative count: "
        f"{len(clean_negatives)} "
        f"!= {EXPECTED_CLEAN_NEG}"
    )


if (
    len(r1_negative_stems)
    != EXPECTED_R1_NEG
):

    raise RuntimeError(
        "Unexpected R1 negative count: "
        f"{len(r1_negative_stems)} "
        f"!= {EXPECTED_R1_NEG}"
    )


if (
    len(unseen_negative_stems)
    != EXPECTED_UNSEEN_NEG
):

    raise RuntimeError(
        "Unexpected unseen negative count: "
        f"{len(unseen_negative_stems)} "
        f"!= {EXPECTED_UNSEEN_NEG}"
    )


# ============================================================
# SAVE IMAGE-PATH MANIFEST
# ============================================================

manifest_path = (
    OUT
    / "fasdd_unseen_negative_train.txt"
)


manifest_path.write_text(
    "\n".join(
        str(
            image_index[
                stem
            ]
        )
        for stem
        in unseen_negative_stems
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# SAVE STEM MANIFEST TOO
# ============================================================

stem_manifest_path = (
    OUT
    / "fasdd_unseen_negative_train_stems.txt"
)


stem_manifest_path.write_text(
    "\n".join(
        unseen_negative_stems
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# CSV
# ============================================================

csv_path = (
    OUT
    / "fasdd_unseen_negative_train.csv"
)


with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stem",
            "image",
            "label",
            "source",
            "category",
        ],
    )

    writer.writeheader()

    for stem in (
        unseen_negative_stems
    ):

        writer.writerow({
            "stem":
                stem,

            "image":
                str(
                    image_index[
                        stem
                    ]
                ),

            "label":
                str(
                    label_index[
                        stem
                    ]
                ),

            "source":
                "FASDD_CLEAN_V1_TRAIN",

            "category":
                "N00_fasdd_unclassified",
        })


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 92)
print("NEGATIVE POOL V1 SUMMARY")
print("=" * 92)

print(
    "CLEAN TRAIN           :",
    len(clean_train_stems)
)

print(
    "CLEAN TRAIN negatives :",
    len(clean_negatives)
)

print(
    "R1 negatives used     :",
    len(r1_negative_stems)
)

print(
    "UNSEEN negatives      :",
    len(unseen_negative_stems)
)

print()

print(
    "POOL ∩ R1 TRAIN       :",
    len(r1_overlap)
)

print(
    "POOL ∩ VAL            :",
    len(val_overlap)
)

print(
    "POOL ∩ TEST           :",
    len(test_overlap)
)

print()

print(
    "Path manifest:",
    manifest_path
)

print(
    "Stem manifest:",
    stem_manifest_path
)

print(
    "CSV          :",
    csv_path
)

print()
print(
    "RESULT: NEGATIVE POOL V1 PASS"
)

print("=" * 92)
