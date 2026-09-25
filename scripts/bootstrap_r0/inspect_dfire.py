#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import json

ROOT = Path("sources/dfire/data")
SPLITS = ["train", "val", "test"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

report = {
    "root": str(ROOT),
    "splits": {},
    "overall": {},
    "issues": {
        "missing_labels": [],
        "orphan_labels": [],
        "malformed_lines": [],
        "invalid_class_ids": [],
        "invalid_coordinates": [],
        "boxes_outside_image": [],
    }
}

overall_categories = Counter()
overall_instances = Counter()

all_image_names = defaultdict(list)


def inspect_label(label_path: Path):
    """
    Returns:
        category
        instance_counter
        issue information
    """

    instance_counter = Counter()
    valid_classes = set()

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()
    except Exception as e:
        report["issues"]["malformed_lines"].append(
            f"{label_path}: cannot read ({e})"
        )
        return "invalid", instance_counter

    # Empty YOLO label = background / negative image
    if not text:
        return "negative", instance_counter

    for line_no, line in enumerate(text.splitlines(), 1):

        parts = line.split()

        if len(parts) != 5:
            report["issues"]["malformed_lines"].append(
                f"{label_path}:{line_no}: {line}"
            )
            continue

        cls_raw, x_raw, y_raw, w_raw, h_raw = parts

        try:
            cls_float = float(cls_raw)

            if not cls_float.is_integer():
                raise ValueError("class id is not integer")

            cls_id = int(cls_float)

            x = float(x_raw)
            y = float(y_raw)
            w = float(w_raw)
            h = float(h_raw)

        except ValueError:
            report["issues"]["malformed_lines"].append(
                f"{label_path}:{line_no}: {line}"
            )
            continue

        # D-Fire should contain only 2 classes
        if cls_id not in (0, 1):
            report["issues"]["invalid_class_ids"].append(
                f"{label_path}:{line_no}: class={cls_id}"
            )
            continue

        valid_classes.add(cls_id)
        instance_counter[cls_id] += 1

        # YOLO normalized xywh validation
        valid_xywh = (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < w <= 1.0
            and 0.0 < h <= 1.0
        )

        if not valid_xywh:
            report["issues"]["invalid_coordinates"].append(
                f"{label_path}:{line_no}: "
                f"{cls_id} {x} {y} {w} {h}"
            )
            continue

        # Check whether resulting box crosses image boundary
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            report["issues"]["boxes_outside_image"].append(
                f"{label_path}:{line_no}: "
                f"{cls_id} {x} {y} {w} {h}"
            )

    if valid_classes == {0}:
        return "class_0_only", instance_counter

    if valid_classes == {1}:
        return "class_1_only", instance_counter

    if valid_classes == {0, 1}:
        return "class_0_and_1", instance_counter

    return "invalid", instance_counter


for split in SPLITS:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    split_categories = Counter()
    split_instances = Counter()

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    labels = sorted(label_dir.rglob("*.txt"))

    # Build relative image stems for orphan-label detection
    image_rel_stems = {
        p.relative_to(image_dir).with_suffix("")
        for p in images
    }

    label_rel_stems = {
        p.relative_to(label_dir).with_suffix("")
        for p in labels
    }

    # Image without label
    for image in images:

        relative = image.relative_to(image_dir)
        label = label_dir / relative.with_suffix(".txt")

        all_image_names[image.name].append(split)

        if not label.exists():
            report["issues"]["missing_labels"].append(str(image))
            split_categories["missing_label"] += 1
            continue

        category, instances = inspect_label(label)

        split_categories[category] += 1
        split_instances.update(instances)

    # Label without image
    orphan_stems = label_rel_stems - image_rel_stems

    for stem in sorted(orphan_stems):
        report["issues"]["orphan_labels"].append(
            str(label_dir / stem.with_suffix(".txt"))
        )

    report["splits"][split] = {
        "images": len(images),
        "labels": len(labels),
        "categories": dict(split_categories),
        "instances": {
            "class_0": split_instances[0],
            "class_1": split_instances[1],
        },
    }

    overall_categories.update(split_categories)
    overall_instances.update(split_instances)


duplicate_filenames = {
    name: splits
    for name, splits in all_image_names.items()
    if len(splits) > 1
}


report["overall"] = {
    "images": sum(
        x["images"]
        for x in report["splits"].values()
    ),
    "labels": sum(
        x["labels"]
        for x in report["splits"].values()
    ),
    "categories": dict(overall_categories),
    "instances": {
        "class_0": overall_instances[0],
        "class_1": overall_instances[1],
    },
    "duplicate_filenames_across_splits":
        len(duplicate_filenames),
}

report["duplicate_filename_samples"] = dict(
    list(duplicate_filenames.items())[:20]
)


# Save full report
output = Path("runs/dfire_inspection.json")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)


# ============================================================
# Console report
# ============================================================

print()
print("=" * 72)
print("D-FIRE DATASET INSPECTION")
print("=" * 72)

for split in SPLITS:

    data = report["splits"][split]

    print(f"\n[{split.upper()}]")
    print(f"Images        : {data['images']}")
    print(f"Labels        : {data['labels']}")

    print("Categories:")

    for category in [
        "class_0_only",
        "class_1_only",
        "class_0_and_1",
        "negative",
        "invalid",
        "missing_label",
    ]:
        print(
            f"  {category:<18}: "
            f"{data['categories'].get(category, 0)}"
        )

    print("Instances:")
    print(
        f"  class_0           : "
        f"{data['instances']['class_0']}"
    )
    print(
        f"  class_1           : "
        f"{data['instances']['class_1']}"
    )


print()
print("=" * 72)
print("OVERALL")
print("=" * 72)

print(f"Images             : {report['overall']['images']}")
print(f"Labels             : {report['overall']['labels']}")

print("\nCategories:")

for category in [
    "class_0_only",
    "class_1_only",
    "class_0_and_1",
    "negative",
    "invalid",
    "missing_label",
]:
    print(
        f"  {category:<18}: "
        f"{overall_categories.get(category, 0)}"
    )

print("\nInstances:")
print(f"  class_0           : {overall_instances[0]}")
print(f"  class_1           : {overall_instances[1]}")

print()
print("Dataset issues:")

for issue_name, items in report["issues"].items():
    print(f"  {issue_name:<24}: {len(items)}")

print(
    "  duplicate filenames       :",
    len(duplicate_filenames)
)

print()
print(f"Full report saved: {output}")

print("=" * 72)
