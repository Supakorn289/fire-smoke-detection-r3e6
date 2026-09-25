from pathlib import Path
import random
import cv2
import numpy as np

random.seed(42)

ROOT = Path("dataset_raw")
OUT = Path("runs/dfire_visual_qa")
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "fire_only": ROOT / "fire_only",
    "smoke_only": ROOT / "smoke_only",
    "fire_smoke": ROOT / "fire_smoke",
    "negative": ROOT / "negative" / "dfire_background",
}

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLES_PER_CATEGORY = 16
CELL_W = 480
CELL_H = 320
GRID_COLS = 4


def draw_labels(image, label_path):
    h, w = image.shape[:2]

    if not label_path.exists():
        return image

    text = label_path.read_text().strip()

    if not text:
        return image

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def make_sheet(category, folder):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    sample_count = min(SAMPLES_PER_CATEGORY, len(images))
    selected = random.sample(images, sample_count)

    cells = []

    for image_path in selected:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        label_path = image_path.with_suffix(".txt")

        image = draw_labels(image, label_path)

        image = cv2.resize(
            image,
            (CELL_W, CELL_H),
            interpolation=cv2.INTER_AREA
        )

        cv2.putText(
            image,
            image_path.name,
            (10, CELL_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cells.append(image)

    while len(cells) % GRID_COLS != 0:
        cells.append(
            np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        )

    rows = []

    for i in range(0, len(cells), GRID_COLS):
        rows.append(
            np.hstack(cells[i:i + GRID_COLS])
        )

    sheet = np.vstack(rows)

    output = OUT / f"{category}_qa.jpg"

    cv2.imwrite(str(output), sheet)

    print(f"{category:15} -> {output}")


print("=" * 60)
print("D-Fire Visual QA")
print("=" * 60)

for category, folder in CATEGORIES.items():
    make_sheet(category, folder)

print("=" * 60)
print("Done")
