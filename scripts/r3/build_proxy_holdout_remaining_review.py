#!/usr/bin/env python3

from pathlib import Path
import csv
import html
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "reports/fasdd/r3_proxy_v1/"
      "proxy_holdout_all.csv"
)

OUT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1/"
      "holdout_remaining_review"
)

CSV_OUT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1/"
      "proxy_holdout_remaining_unverified.csv"
)

EXPECTED_ALL = 406
EXPECTED_ALREADY_REVIEWED = 109
EXPECTED_REMAINING = 297
THRESHOLD = 0.25


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)


with SOURCE.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != EXPECTED_ALL:
    raise RuntimeError(
        f"Expected {EXPECTED_ALL} holdout images, "
        f"got {len(rows)}"
    )


already_reviewed = [
    row for row in rows
    if float(row["top_conf"]) >= THRESHOLD
]

remaining = [
    row for row in rows
    if float(row["top_conf"]) < THRESHOLD
]


if len(already_reviewed) != EXPECTED_ALREADY_REVIEWED:

    raise RuntimeError(
        "Expected "
        f"{EXPECTED_ALREADY_REVIEWED} previously "
        "reviewed holdout alerts, got "
        f"{len(already_reviewed)}"
    )


if len(remaining) != EXPECTED_REMAINING:

    raise RuntimeError(
        f"Expected {EXPECTED_REMAINING} remaining, "
        f"got {len(remaining)}"
    )


if OUT.exists():
    shutil.rmtree(OUT)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# Stronger R2.1 predictions first,
# so the most suspicious remaining images
# appear at the top of the gallery.
remaining.sort(
    key=lambda row:
        float(row["top_conf"]),
    reverse=True,
)


fieldnames = list(
    remaining[0].keys()
)

if "human_decision" not in fieldnames:
    fieldnames.append(
        "human_decision"
    )

if "human_note" not in fieldnames:
    fieldnames.append(
        "human_note"
    )


with CSV_OUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in remaining:

        row = dict(row)

        row["human_decision"] = ""
        row["human_note"] = ""

        writer.writerow(row)


cards = []


for i, row in enumerate(
    remaining,
    start=1,
):

    source = Path(
        row["proxy_path"]
    )

    if not source.exists():
        raise FileNotFoundError(source)


    suffix = source.suffix.lower()

    filename = (
        f"{i:04d}_"
        f"cam{row['camera_id']}_"
        f"{row['top_class']}_"
        f"{float(row['top_conf']):.3f}"
        f"{suffix}"
    )


    destination = (
        OUT / filename
    )


    os.symlink(
        source.resolve(),
        destination,
    )


    cards.append(
        f"""
        <div class="card">
            <div class="title">
                #{i:04d}
                | cam {html.escape(str(row['camera_id']))}
                | {html.escape(str(row['top_class']))}
                {float(row['top_conf']):.3f}
            </div>

            <img
                src="{html.escape(filename)}"
                loading="lazy"
            >

            <div class="meta">
                fire={float(row['fire_conf']):.3f}
                &nbsp;
                smoke={float(row['smoke_conf']):.3f}
            </div>
        </div>
        """
    )


page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">

<title>Proxy Holdout Remaining Review</title>

<style>
body {{
    margin: 20px;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
}}

.notice {{
    background: #222;
    border: 1px solid #555;
    padding: 14px;
    margin-bottom: 20px;
    line-height: 1.5;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fill, minmax(330px, 1fr));
    gap: 15px;
}}

.card {{
    background: #1c1c1c;
    border: 1px solid #444;
}}

.title {{
    padding: 8px;
    font-weight: bold;
}}

.card img {{
    display: block;
    width: 100%;
    height: 270px;
    object-fit: contain;
    background: #000;
}}

.meta {{
    padding: 8px;
    font-family: monospace;
}}
</style>
</head>

<body>

<h1>Proxy Holdout — Remaining Review</h1>

<div class="notice">
Total remaining: {len(remaining)} images<br>
These are the holdout images that R2.1 scored below 0.25.<br><br>

Verify only one thing:
<b>Does the image contain real fire or real smoke?</b><br><br>

If none do, report: ALL 297 NEGATIVE.<br>
If any do, record their image numbers and do not classify them as negative.
</div>

<div class="grid">
{''.join(cards)}
</div>

</body>
</html>
"""


index = (
    OUT / "index.html"
)

index.write_text(
    page,
    encoding="utf-8",
)


print("=" * 90)
print("PROXY HOLDOUT REMAINING REVIEW")
print("=" * 90)

print(
    "Holdout total          :",
    len(rows)
)

print(
    "Previously reviewed    :",
    len(already_reviewed)
)

print(
    "Remaining to review    :",
    len(remaining)
)

print()

print(
    "Gallery:",
    index
)

print(
    "CSV    :",
    CSV_OUT
)

print()

print(
    "RESULT: HOLDOUT REVIEW GALLERY PASS"
)

print("=" * 90)
