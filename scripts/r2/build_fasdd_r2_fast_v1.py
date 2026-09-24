#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

R1 = (
    ROOT
    / "datasets/fasdd_r1_v1"
)

R2 = (
    ROOT
    / "datasets/fasdd_r2_fast_v1"
)

HARDNEG_DIR = (
    ROOT
    / "reports/negative_v1/final_v1"
)

HARDNEG_TRAIN = (
    HARDNEG_DIR
    / "hardneg_train.csv"
)

HARDNEG_VAL = (
    HARDNEG_DIR
    / "hardneg_val.csv"
)

CURATED_ROOT = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def symlink_file(
    source,
    destination,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        or destination.is_symlink()
    ):
        destination.unlink()

    os.symlink(
        source.resolve(),
        destination,
    )


def copy_r1_split(
    split,
):

    source_images = (
        R1
        / "images"
        / split
    )

    source_labels = (
        R1
        / "labels"
        / split
    )

    if not source_images.exists():
        raise FileNotFoundError(
            source_images
        )

    if not source_labels.exists():
        raise FileNotFoundError(
            source_labels
        )


    images = sorted(
        p
        for p
        in source_images.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTS
        )
    )


    count = 0

    for image in images:

        rel = image.relative_to(
            source_images
        )

        label_rel = (
            rel.with_suffix(
                ".txt"
            )
        )

        label = (
            source_labels
            / label_rel
        )

        if not label.exists():
            raise FileNotFoundError(
                label
            )


        symlink_file(
            image,
            R2
            / "images"
            / split
            / rel,
        )

        symlink_file(
            label,
            R2
            / "labels"
            / split
            / label_rel,
        )

        count += 1

    return count


def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def build_label_index(
    wanted_stems,
):

    index = {}

    for path in (
        CURATED_ROOT.rglob(
            "*.txt"
        )
    ):

        if (
            path.stem
            not in wanted_stems
        ):
            continue

        if path.stem in index:
            raise RuntimeError(
                "Duplicate label stem: "
                f"{path.stem}"
            )

        index[
            path.stem
        ] = path.resolve()

    missing = (
        wanted_stems
        - set(index)
    )

    if missing:
        raise RuntimeError(
            "Missing labels: "
            f"{sorted(missing)[:20]}"
        )

    return index


def add_hardneg_split(
    rows,
    split_name,
    label_index,
):

    count = 0

    for row in rows:

        image = Path(
            row["image"]
        )

        if not image.exists():
            raise FileNotFoundError(
                image
            )

        stem = image.stem

        label = label_index[
            stem
        ]


        # These MUST remain empty labels.
        if label.read_text(
            encoding="utf-8"
        ).strip():

            raise RuntimeError(
                f"Hard negative has "
                f"non-empty label: {label}"
            )


        symlink_file(
            image,
            R2
            / "images"
            / split_name
            / "hardneg_v1"
            / image.name,
        )


        symlink_file(
            label,
            R2
            / "labels"
            / split_name
            / "hardneg_v1"
            / f"{stem}.txt",
        )

        count += 1

    return count


def count_images(path):

    return sum(
        1
        for p in path.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTS
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = parser.parse_args()


    if R2.exists():

        if not args.rebuild:

            raise RuntimeError(
                f"{R2} already exists.\n"
                "Use --rebuild to recreate."
            )

        shutil.rmtree(
            R2
        )


    R2.mkdir(
        parents=True,
        exist_ok=True,
    )


    train_hn = read_csv(
        HARDNEG_TRAIN
    )

    val_hn = read_csv(
        HARDNEG_VAL
    )


    train_hn_stems = {
        Path(row["image"]).stem
        for row in train_hn
    }

    val_hn_stems = {
        Path(row["image"]).stem
        for row in val_hn
    }


    if (
        train_hn_stems
        & val_hn_stems
    ):
        raise RuntimeError(
            "Hard-negative train/val leakage"
        )


    wanted = (
        train_hn_stems
        | val_hn_stems
    )


    print("=" * 92)
    print(
        "BUILD FASDD R2 FAST V1"
    )
    print("=" * 92)


    print(
        "Copying R1 TRAIN..."
    )

    r1_train_count = (
        copy_r1_split(
            "train"
        )
    )


    print(
        "Copying R1 VAL..."
    )

    r1_val_count = (
        copy_r1_split(
            "val"
        )
    )


    print(
        "Indexing curated labels..."
    )

    label_index = (
        build_label_index(
            wanted
        )
    )


    print(
        "Adding verified "
        "hard-negative TRAIN..."
    )

    hardneg_train_count = (
        add_hardneg_split(
            train_hn,
            "train",
            label_index,
        )
    )


    print(
        "Building held-out "
        "hard-negative VAL..."
    )

    hardneg_val_count = (
        add_hardneg_split(
            val_hn,
            "hardneg_val",
            label_index,
        )
    )


    total_train = count_images(
        R2
        / "images/train"
    )

    total_val = count_images(
        R2
        / "images/val"
    )

    total_hardneg_val = count_images(
        R2
        / "images/hardneg_val"
    )


    expected_train = (
        r1_train_count
        + hardneg_train_count
    )


    if (
        total_train
        != expected_train
    ):
        raise RuntimeError(
            "R2 train count mismatch"
        )


    if total_val != r1_val_count:
        raise RuntimeError(
            "Main VAL count mismatch"
        )


    if (
        total_hardneg_val
        != hardneg_val_count
    ):
        raise RuntimeError(
            "Hard-negative VAL "
            "count mismatch"
        )


    # ========================================================
    # YAML
    # ========================================================

    data_yaml = (
        R2
        / "data.yaml"
    )


    data_yaml.write_text(
        "\n".join([
            f"path: {R2}",
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


    print()
    print("=" * 92)
    print(
        "R2 FAST V1 SUMMARY"
    )
    print("=" * 92)

    print(
        "R1 train              :",
        r1_train_count
    )

    print(
        "New hardneg train     :",
        hardneg_train_count
    )

    print(
        "R2 train total        :",
        total_train
    )

    print(
        "Main VAL              :",
        total_val
    )

    print(
        "Held-out hardneg VAL  :",
        total_hardneg_val
    )

    print()
    print(
        "Dataset:",
        R2
    )

    print(
        "YAML   :",
        data_yaml
    )

    print()
    print(
        "RESULT: FASDD R2 FAST V1 PASS"
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
