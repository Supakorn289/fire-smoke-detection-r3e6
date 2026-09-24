#!/usr/bin/env python3

from pathlib import Path
import csv
import html
import os


ROOT = Path(__file__).resolve().parents[2]

MINED = (
    ROOT
    / "reports/negative_v1/mining/"
    "fasdd_unseen_negative_mined.csv"
)

OUT = (
    ROOT
    / "reports/negative_v1/review_v1"
)

IMAGES_OUT = (
    OUT / "images"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGES_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_rows():
    if not MINED.exists():
        raise FileNotFoundError(MINED)

    with MINED.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    return rows


rows = load_rows()

selected = [
    row
    for row in rows
    if row["severity"]
    in {
        "EXTREME",
        "HARD",
    }
]

selected.sort(
    key=lambda r:
        float(
            r["hardness_score"]
        ),
    reverse=True,
)


print("=" * 90)
print("HARD NEGATIVE REVIEW V1")
print("=" * 90)

print(
    "Mined total:",
    len(rows)
)

print(
    "Review candidates:",
    len(selected)
)


# ============================================================
# Create symlinks
# ============================================================

for index, row in enumerate(
    selected,
    start=1,
):
    source = Path(
        row["image"]
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    name = (
        f"{index:04d}_"
        f"{row['severity']}_"
        f"{source.name}"
    )

    dest = (
        IMAGES_OUT / name
    )

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    os.symlink(
        source,
        dest,
    )

    row[
        "review_image"
    ] = str(dest)

    row[
        "review_index"
    ] = index

    # Human decision:
    #
    # APPROVE_NEGATIVE
    # RELABEL_FIRE
    # RELABEL_SMOKE
    # RELABEL_FIRE_SMOKE
    # QUARANTINE
    #
    row[
        "human_decision"
    ] = ""

    row[
        "human_note"
    ] = ""


# ============================================================
# Review CSV
# ============================================================

csv_path = (
    OUT
    / "hard_negative_review_v1.csv"
)

fields = list(
    selected[0].keys()
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        selected
    )


# ============================================================
# HTML gallery
# ============================================================

cards = []

for row in selected:

    image_path = Path(
        row["review_image"]
    )

    relative = (
        "images/"
        + image_path.name
    )

    card = f"""
    <div class="card">
        <div class="number">
            #{int(row['review_index']):04d}
        </div>

        <img
            src="{html.escape(relative)}"
            loading="lazy"
        >

        <div class="meta">
            <b>{html.escape(row['severity'])}</b><br>

            score =
            {float(row['hardness_score']):.4f}
            <br><br>

            N =
            {float(row['n_top_conf']):.3f}
            ({html.escape(row['n_top_class'])})
            <br>

            S =
            {float(row['s_top_conf']):.3f}
            ({html.escape(row['s_top_class'])})
            <br>

            M =
            {float(row['m_top_conf']):.3f}
            ({html.escape(row['m_top_class'])})
            <br><br>

            consensus =
            <b>
            {html.escape(row['consensus_class'])}
            </b>

            <br><br>

            {html.escape(Path(row['image']).name)}
        </div>
    </div>
    """

    cards.append(card)


page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">

<title>
Hard Negative Review V1
</title>

<style>

body {{
    font-family:
        Arial,
        sans-serif;

    background:
        #111;

    color:
        #eee;

    margin:
        20px;
}}

h1 {{
    margin-bottom:
        4px;
}}

.info {{
    color:
        #aaa;

    margin-bottom:
        20px;
}}

.grid {{
    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(
                320px,
                1fr
            )
        );

    gap:
        16px;
}}

.card {{
    background:
        #1d1d1d;

    border:
        1px solid #444;

    border-radius:
        8px;

    overflow:
        hidden;
}}

.card img {{
    display:
        block;

    width:
        100%;

    height:
        250px;

    object-fit:
        contain;

    background:
        #000;
}}

.number {{
    padding:
        8px;

    background:
        #292929;

    font-weight:
        bold;
}}

.meta {{
    padding:
        12px;

    line-height:
        1.45;

    font-family:
        monospace;
}}

</style>
</head>

<body>

<h1>
Hard Negative Review V1
</h1>

<div class="info">
472 Extreme/Hard candidates.
ตรวจว่าภาพไม่มี Fire/Smoke จริงก่อนนำเข้า R2
</div>

<div class="grid">
{''.join(cards)}
</div>

</body>
</html>
"""


html_path = (
    OUT
    / "index.html"
)

html_path.write_text(
    page,
    encoding="utf-8",
)


print()
print(
    "Review CSV:",
    csv_path
)

print(
    "HTML      :",
    html_path
)

print(
    "Images    :",
    IMAGES_OUT
)

print()
print(
    "RESULT: HARD NEGATIVE REVIEW V1 READY"
)

print("=" * 90)
