#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import shutil

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "dataset_raw"
REVIEW = ROOT / "reports/human_reference/dfire_human_qa_200.csv"
OUT = ROOT / "datasets/bootstrap_v1"
REPORT = ROOT / "reports/bootstrap_v1/manifest.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

CATEGORIES = {
    "fire_only": RAW / "fire_only",
    "smoke_only": RAW / "smoke_only",
    "fire_smoke": RAW / "fire_smoke",
    "negative": RAW / "negative/dfire_background",
}


def source_split(path: Path):
    # dfire_train_xxx.jpg
    parts = path.stem.split("_", 2)

    if len(parts) >= 3 and parts[0] == "dfire":
        s = parts[1]

        if s in {"train", "val", "test"}:
            return s

    return None


def images_in(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


def load_human_qa():
    rows = list(
        csv.DictReader(
            REVIEW.open(encoding="utf-8")
        )
    )

    result = {}

    for row in rows:
        result[Path(row["image"]).resolve()] = row

    return result


def should_include(category, image, qa):

    # QA ของเราพบว่า mixed + negative เชื่อถือได้สูง
    if category in {"fire_smoke", "negative"}:
        return True, "high_confidence_category"

    # fire_only / smoke_only:
    # Bootstrap รอบแรกเอาเฉพาะที่คน APPROVE
    row = qa.get(image.resolve())

    if row and row["status"] == "approve":
        return True, "human_approved"

    return False, "not_bootstrap_safe"


def main():

    qa = load_human_qa()

    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    records = []
    stats = Counter()

    for category, folder in CATEGORIES.items():

        for image in images_in(folder):

            split = source_split(image)

            if split is None:
                stats["unknown_split"] += 1
                continue

            include, reason = should_include(
                category,
                image,
                qa
            )

            if not include:
                stats[f"excluded_{category}"] += 1
                continue

            dst_img = OUT / "images" / split / image.name
            dst_lbl = OUT / "labels" / split / f"{image.stem}.txt"

            shutil.copy2(image, dst_img)

            src_lbl = image.with_suffix(".txt")

            if category == "negative":
                # Negative image -> YOLO empty annotation
                dst_lbl.write_text("", encoding="utf-8")
            else:
                if not src_lbl.exists():
                    stats["missing_positive_label"] += 1
                    dst_img.unlink(missing_ok=True)
                    continue

                shutil.copy2(src_lbl, dst_lbl)

            records.append({
                "image": image.name,
                "category": category,
                "split": split,
                "selection_reason": reason,
                "source": "dfire",
            })

            stats[f"included_{category}"] += 1
            stats[f"split_{split}"] += 1

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REPORT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "category",
                "split",
                "selection_reason",
                "source",
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    yaml = OUT / "data.yaml"

    yaml.write_text(
        f"""path: {OUT}

train: images/train
val: images/val
test: images/test

names:
  0: fire
  1: smoke
""",
        encoding="utf-8"
    )

    print("=" * 65)
    print("BOOTSTRAP DATASET V1")
    print("=" * 65)

    for cat in (
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative"
    ):
        print(
            f"{cat:15}: "
            f"{stats[f'included_{cat}']}"
        )

    print("-" * 65)

    for split in ("train", "val", "test"):
        print(
            f"{split:15}: "
            f"{stats[f'split_{split}']}"
        )

    print("-" * 65)
    print(f"TOTAL          : {len(records)}")
    print(f"Manifest       : {REPORT}")
    print(f"Dataset YAML   : {yaml}")
    print("=" * 65)


if __name__ == "__main__":
    main()
