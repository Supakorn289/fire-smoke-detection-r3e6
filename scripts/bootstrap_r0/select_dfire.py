from pathlib import Path
import random
import shutil
import csv

ROOT = Path("sources/dfire/data")
OUT = Path("dataset_raw")
SEED = 42

# D-Fire: 0=smoke, 1=fire
# Project: 0=fire, 1=smoke

TARGETS = {
    "train": {"fire_only":455, "smoke_only":455, "fire_smoke":350, "negative":140},
    "val":   {"fire_only":130, "smoke_only":130, "fire_smoke":100, "negative":40},
    "test":  {"fire_only":65,  "smoke_only":65,  "fire_smoke":50,  "negative":20},
}

DEST = {
    "fire_only": OUT / "fire_only",
    "smoke_only": OUT / "smoke_only",
    "fire_smoke": OUT / "fire_smoke",
    "negative": OUT / "negative" / "dfire_background",
}

for p in DEST.values():
    p.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def read_valid_label(path):
    text = path.read_text().strip()

    if not text:
        return "negative", []

    rows = []
    classes = set()

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            return None, None

        try:
            c = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            return None, None

        if c not in (0, 1):
            return None, None

        if not (
            0 <= x <= 1 and
            0 <= y <= 1 and
            0 < w <= 1 and
            0 < h <= 1
        ):
            return None, None

        if (
            x - w/2 < 0 or
            y - h/2 < 0 or
            x + w/2 > 1 or
            y + h/2 > 1
        ):
            return None, None

        classes.add(c)
        rows.append((c, x, y, w, h))

    if classes == {1}:
        category = "fire_only"
    elif classes == {0}:
        category = "smoke_only"
    elif classes == {0, 1}:
        category = "fire_smoke"
    else:
        return None, None

    return category, rows


manifest = []

for split, wanted in TARGETS.items():

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    pools = {k: [] for k in wanted}

    for image in sorted(image_dir.glob("*")):

        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        category, rows = read_valid_label(label)

        if category in pools:
            pools[category].append((image, rows))

    for category, amount in wanted.items():

        candidates = pools[category]

        if len(candidates) < amount:
            raise RuntimeError(
                f"{split}/{category}: "
                f"need {amount}, have {len(candidates)}"
            )

        selected = random.sample(candidates, amount)

        for image, rows in selected:

            new_stem = f"dfire_{split}_{image.stem}"
            dst_image = DEST[category] / f"{new_stem}{image.suffix.lower()}"

            shutil.copy2(image, dst_image)

            # Negative = no label file
            if category != "negative":

                dst_label = DEST[category] / f"{new_stem}.txt"

                with dst_label.open("w") as f:

                    for c, x, y, w, h in rows:

                        # Remap D-Fire -> Project
                        # smoke 0 -> 1
                        # fire  1 -> 0
                        new_c = 1 - c

                        f.write(
                            f"{new_c} "
                            f"{x:.8f} {y:.8f} "
                            f"{w:.8f} {h:.8f}\n"
                        )

            manifest.append([
                "dfire",
                split,
                category,
                str(image),
                str(dst_image),
            ])


manifest_path = OUT / "dfire_manifest.csv"

with manifest_path.open("w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "source",
        "source_split",
        "category",
        "source_image",
        "output_image",
    ])

    writer.writerows(manifest)


print("=" * 60)
print("D-Fire selection completed")
print("=" * 60)

for category in DEST:

    count = sum(
        1 for row in manifest
        if row[2] == category
    )

    print(f"{category:15}: {count}")

print("-" * 60)
print("TOTAL          :", len(manifest))
print("Manifest       :", manifest_path)
