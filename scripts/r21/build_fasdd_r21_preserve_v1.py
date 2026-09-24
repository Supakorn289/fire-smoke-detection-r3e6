#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import hashlib
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "datasets/fasdd_r2_fast_v1"
)

R1 = (
    ROOT
    / "datasets/fasdd_r1_v1"
)

OUT = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1"
)

REPORT = (
    ROOT
    / "reports/fasdd/r21_preserve_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)

SMALL_AREA = 0.005

EXPECTED_BASE_TRAIN = 12378
EXPECTED_VAL = 3000

SMALL_FIRE_SAMPLE = 1000
GENERAL_FIRE_SAMPLE = 500

SMALL_SMOKE_REPEATS = 2

SEED = 42

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def stable_key(stem, tag):
    text = (
        f"{SEED}:{tag}:{stem}"
    ).encode("utf-8")

    return hashlib.sha256(
        text
    ).hexdigest()


def image_index(root):
    result = {}

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        if p.stem in result:
            raise RuntimeError(
                f"Duplicate image stem: {p.stem}"
            )

        result[p.stem] = p.resolve()

    return result


def label_index(root):
    result = {}

    for p in root.rglob("*.txt"):

        if p.stem in result:
            raise RuntimeError(
                f"Duplicate label stem: {p.stem}"
            )

        result[p.stem] = p.resolve()

    return result


def parse_label(path):
    objects = []

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return objects

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Invalid YOLO row: {path}\n{line}"
            )

        cls = int(float(parts[0]))
        w = float(parts[3])
        h = float(parts[4])

        objects.append(
            (
                cls,
                w * h,
            )
        )

    return objects


def symlink(source, destination):
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


def count_images(root):
    return sum(
        1
        for p in root.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
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

    if not BASE.exists():
        raise FileNotFoundError(BASE)

    if not R1.exists():
        raise FileNotFoundError(R1)

    if OUT.exists():

        if not args.rebuild:
            raise RuntimeError(
                f"{OUT} already exists. "
                "Use --rebuild."
            )

        shutil.rmtree(
            OUT
        )

    print("=" * 96)
    print(
        "BUILD FASDD R2.1 PRESERVE-POSITIVE V1"
    )
    print("=" * 96)

    print(
        "Copying R2 base dataset..."
    )

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

    # IMPORTANT:
    # remove copied Ultralytics caches because
    # replay samples will change the dataset.
    for cache in OUT.rglob(
        "*.cache"
    ):
        cache.unlink()

    base_train = count_images(
        OUT / "images/train"
    )

    val_count = count_images(
        OUT / "images/val"
    )

    if base_train != EXPECTED_BASE_TRAIN:
        raise RuntimeError(
            f"Base train changed: "
            f"{base_train} != "
            f"{EXPECTED_BASE_TRAIN}"
        )

    if val_count != EXPECTED_VAL:
        raise RuntimeError(
            f"VAL changed: "
            f"{val_count} != "
            f"{EXPECTED_VAL}"
        )

    r1_images = image_index(
        R1 / "images/train"
    )

    r1_labels = label_index(
        R1 / "labels/train"
    )

    if set(r1_images) != set(r1_labels):
        raise RuntimeError(
            "R1 image/label stem mismatch"
        )

    small_smoke = []
    small_fire = []
    general_fire = []

    for stem, label in r1_labels.items():

        objects = parse_label(
            label
        )

        has_fire = any(
            cls == 0
            for cls, area
            in objects
        )

        has_small_fire = any(
            cls == 0
            and area < SMALL_AREA
            for cls, area
            in objects
        )

        has_small_smoke = any(
            cls == 1
            and area < SMALL_AREA
            for cls, area
            in objects
        )

        if has_small_smoke:
            small_smoke.append(
                stem
            )

        elif has_small_fire:
            small_fire.append(
                stem
            )

        elif has_fire:
            general_fire.append(
                stem
            )

    small_smoke.sort(
        key=lambda s:
            stable_key(
                s,
                "small_smoke",
            )
    )

    small_fire.sort(
        key=lambda s:
            stable_key(
                s,
                "small_fire",
            )
    )

    general_fire.sort(
        key=lambda s:
            stable_key(
                s,
                "general_fire",
            )
    )

    print()
    print(
        "Small-smoke images available:",
        len(small_smoke)
    )

    print(
        "Small-fire candidates:",
        len(small_fire)
    )

    print(
        "General-fire candidates:",
        len(general_fire)
    )

    # R1 was intentionally built to contain
    # all 333 train images with small smoke.
    if len(small_smoke) != 333:
        raise RuntimeError(
            "Expected 333 small-smoke "
            f"training images, got "
            f"{len(small_smoke)}"
        )

    if len(small_fire) < SMALL_FIRE_SAMPLE:
        raise RuntimeError(
            "Not enough small-fire images"
        )

    if len(general_fire) < GENERAL_FIRE_SAMPLE:
        raise RuntimeError(
            "Not enough general-fire images"
        )

    selected_small_fire = (
        small_fire[
            :SMALL_FIRE_SAMPLE
        ]
    )

    selected_general_fire = (
        general_fire[
            :GENERAL_FIRE_SAMPLE
        ]
    )

    replay_rows = []

    def add_replay(
        stem,
        prefix,
        group,
    ):
        source_image = (
            r1_images[
                stem
            ]
        )

        source_label = (
            r1_labels[
                stem
            ]
        )

        new_stem = (
            f"{prefix}_{stem}"
        )

        image_dest = (
            OUT
            / "images/train/"
              "replay_preserve_v1"
            / (
                new_stem
                + source_image.suffix.lower()
            )
        )

        label_dest = (
            OUT
            / "labels/train/"
              "replay_preserve_v1"
            / f"{new_stem}.txt"
        )

        symlink(
            source_image,
            image_dest,
        )

        symlink(
            source_label,
            label_dest,
        )

        replay_rows.append({
            "source_stem":
                stem,

            "replay_stem":
                new_stem,

            "group":
                group,

            "source_image":
                str(source_image),

            "source_label":
                str(source_label),
        })

    # --------------------------------------------------------
    # Small Smoke:
    # two additional replay copies of every image
    # --------------------------------------------------------

    for repeat in range(
        1,
        SMALL_SMOKE_REPEATS + 1,
    ):
        for stem in small_smoke:

            add_replay(
                stem,
                f"ss{repeat}",
                "small_smoke",
            )

    # --------------------------------------------------------
    # Small Fire:
    # one additional replay copy
    # --------------------------------------------------------

    for stem in selected_small_fire:

        add_replay(
            stem,
            "sf",
            "small_fire",
        )

    # --------------------------------------------------------
    # General Fire:
    # one replay copy
    # --------------------------------------------------------

    for stem in selected_general_fire:

        add_replay(
            stem,
            "gf",
            "general_fire",
        )

    replay_count = len(
        replay_rows
    )

    final_train = count_images(
        OUT / "images/train"
    )

    expected_replay = (
        333 * SMALL_SMOKE_REPEATS
        + SMALL_FIRE_SAMPLE
        + GENERAL_FIRE_SAMPLE
    )

    expected_train = (
        EXPECTED_BASE_TRAIN
        + expected_replay
    )

    if replay_count != expected_replay:
        raise RuntimeError(
            "Replay count mismatch"
        )

    if final_train != expected_train:
        raise RuntimeError(
            f"Final train mismatch: "
            f"{final_train} != "
            f"{expected_train}"
        )

    # Remove any cache one more time
    # after replay creation.
    for cache in OUT.rglob(
        "*.cache"
    ):
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

    csv_path = (
        REPORT
        / "replay_selection.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_stem",
                "replay_stem",
                "group",
                "source_image",
                "source_label",
            ],
        )

        writer.writeheader()
        writer.writerows(
            replay_rows
        )

    print()
    print("=" * 96)
    print(
        "R2.1 DATASET SUMMARY"
    )
    print("=" * 96)

    print(
        "R2 base train       :",
        base_train
    )

    print(
        "Small-smoke replay  :",
        333
        * SMALL_SMOKE_REPEATS
    )

    print(
        "Small-fire replay   :",
        SMALL_FIRE_SAMPLE
    )

    print(
        "General-fire replay :",
        GENERAL_FIRE_SAMPLE
    )

    print(
        "Replay total        :",
        replay_count
    )

    print(
        "R2.1 train total    :",
        final_train
    )

    print(
        "Main VAL            :",
        val_count
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

    print(
        "Replay :",
        csv_path
    )

    print()
    print(
        "RESULT: FASDD R2.1 "
        "PRESERVE V1 PASS"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
