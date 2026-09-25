#!/usr/bin/env python3

from pathlib import Path
import shutil
import csv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bootstrap_v1"

BACKUP = ROOT / "datasets/bootstrap_v1_labels_backup"
REPORT = ROOT / "reports/bootstrap_v1/box_sanitize_report.csv"

SPLITS = ["train", "val", "test"]

report = []


def clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


for split in SPLITS:

    label_dir = DATA / "labels" / split
    backup_dir = BACKUP / split

    backup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for label_path in sorted(label_dir.glob("*.txt")):

        # Backup original ครั้งแรกเท่านั้น
        backup_path = backup_dir / label_path.name

        if not backup_path.exists():
            shutil.copy2(
                label_path,
                backup_path
            )

        text = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        if not text:
            continue

        new_lines = []
        changed = False

        for line_no, line in enumerate(
            text.splitlines(),
            start=1
        ):

            parts = line.split()

            if len(parts) != 5:
                new_lines.append(line)
                continue

            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(
                    float,
                    parts[1:]
                )
            except ValueError:
                new_lines.append(line)
                continue

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            outside = (
                x1 < 0 or
                y1 < 0 or
                x2 > 1 or
                y2 > 1
            )

            if not outside:
                new_lines.append(line)
                continue

            # Clip ขอบ bbox ให้ตรงกับขอบภาพ
            cx1 = clip(x1)
            cy1 = clip(y1)
            cx2 = clip(x2)
            cy2 = clip(y2)

            new_w = cx2 - cx1
            new_h = cy2 - cy1

            # ถ้าหลัง clip ไม่มีพื้นที่เหลือ
            # ให้ drop bbox นี้
            if new_w <= 0 or new_h <= 0:

                report.append({
                    "split": split,
                    "label": label_path.name,
                    "line": line_no,
                    "class": cls,
                    "action": "drop_zero_after_clip",
                    "old_box": line,
                    "new_box": "",
                })

                changed = True
                continue

            new_x = (cx1 + cx2) / 2
            new_y = (cy1 + cy2) / 2

            new_line = (
                f"{cls} "
                f"{new_x:.10f} "
                f"{new_y:.10f} "
                f"{new_w:.10f} "
                f"{new_h:.10f}"
            )

            new_lines.append(new_line)

            report.append({
                "split": split,
                "label": label_path.name,
                "line": line_no,
                "class": cls,
                "action": "clip",
                "old_box": line,
                "new_box": new_line,
            })

            changed = True

        if changed:

            label_path.write_text(
                "\n".join(new_lines)
                + ("\n" if new_lines else ""),
                encoding="utf-8"
            )


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
            "split",
            "label",
            "line",
            "class",
            "action",
            "old_box",
            "new_box",
        ]
    )

    writer.writeheader()
    writer.writerows(report)


print("=" * 65)
print("BOOTSTRAP BOX SANITIZER")
print("=" * 65)

print(f"Boxes modified : {len(report)}")

clips = sum(
    r["action"] == "clip"
    for r in report
)

drops = sum(
    r["action"] == "drop_zero_after_clip"
    for r in report
)

print(f"Clipped        : {clips}")
print(f"Dropped        : {drops}")
print()
print(f"Backup         : {BACKUP}")
print(f"Report         : {REPORT}")
print("=" * 65)
