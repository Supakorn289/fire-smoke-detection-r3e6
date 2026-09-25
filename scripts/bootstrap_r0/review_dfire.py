#!/usr/bin/env python3

from pathlib import Path
from flask import Flask, Response, redirect, render_template_string, request
import csv
import random
import cv2

ROOT = Path.cwd()

QUEUE_FILE = ROOT / "dataset_review" / "review_queue.csv"

SOURCES = {
    "fire_only": ROOT / "dataset_raw" / "fire_only",
    "smoke_only": ROOT / "dataset_raw" / "smoke_only",
    "fire_smoke": ROOT / "dataset_raw" / "fire_smoke",
    "negative": ROOT / "dataset_raw" / "negative" / "dfire_background",
}

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLE_PER_CATEGORY = 50
SEED = 20260808

FIELDS = [
    "id",
    "category",
    "source_split",
    "image",
    "label",
    "status",
    "note",
]

app = Flask(__name__)


def build_queue():
    if QUEUE_FILE.exists():
        return

    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    rows = []

    for category, folder in SOURCES.items():

        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        selected = rng.sample(
            images,
            min(SAMPLE_PER_CATEGORY, len(images))
        )

        for image in selected:

            # filename เช่น dfire_train_WEB01234.jpg
            parts = image.stem.split("_", 2)

            if len(parts) >= 3 and parts[0] == "dfire":
                source_split = parts[1]
            else:
                source_split = "unknown"

            label = image.with_suffix(".txt")

            rows.append({
                "id": str(len(rows)),
                "category": category,
                "source_split": source_split,
                "image": str(image),
                "label": str(label) if label.exists() else "",
                "status": "unreviewed",
                "note": "",
            })

    with QUEUE_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_queue():
    build_queue()

    with QUEUE_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def save_queue(rows):
    with QUEUE_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def draw_boxes(image, label_path):

    if not label_path or not Path(label_path).exists():
        return image

    h, w = image.shape[:2]

    text = Path(label_path).read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()

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

        name = CLASS_NAMES.get(
            cls_id,
            f"class_{cls_id}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            name,
            (x1, max(25, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return image


HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>D-Fire QA</title>

<style>
body {
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}

img {
    width: 100%;
    max-height: 720px;
    object-fit: contain;
    background: #000;
}

.card {
    background: #1d1d1d;
    padding: 16px;
    border-radius: 10px;
}

.info {
    margin-bottom: 12px;
}

button {
    padding: 12px 20px;
    margin: 5px;
    font-size: 16px;
    cursor: pointer;
}

input {
    width: 95%;
    padding: 10px;
    margin: 10px 0;
}

.approve { background:#3fae58; }
.relabel { background:#d5a632; }
.quarantine { background:#d9534f; }
.skip { background:#777; }

a {
    color: #8ac7ff;
}
</style>
</head>

<body>

<h1>D-Fire Dataset Review</h1>

<div class="card">

<div class="info">
<b>Progress:</b>
{{ reviewed }} / {{ total }}
<br>

<b>Index:</b>
{{ index + 1 }} / {{ total }}
<br>

<b>Category:</b>
{{ row.category }}
<br>

<b>Source split:</b>
{{ row.source_split }}
<br>

<b>File:</b>
{{ row.image }}
<br>

<b>Current status:</b>
{{ row.status }}
</div>

<img src="/image/{{ index }}">

<form method="post" action="/mark/{{ index }}">

<input
    name="note"
    placeholder="Note เช่น missing smoke / bad smoke box / ambiguous"
    value="{{ row.note }}"
>

<br>

<button
    class="approve"
    name="status"
    value="approve">
APPROVE
</button>

<button
    class="relabel"
    name="status"
    value="relabel">
RELABEL
</button>

<button
    class="quarantine"
    name="status"
    value="quarantine">
QUARANTINE
</button>

<button
    class="skip"
    name="status"
    value="skip">
SKIP
</button>

</form>

<p>
<a href="/?i={{ prev_index }}">← Previous</a>
&nbsp;&nbsp;
<a href="/?i={{ next_index }}">Next →</a>
</p>

</div>


<p>
<b>Keyboard:</b>
A = Approve |
R = Relabel |
Q = Quarantine |
S = Skip
</p>

<script id="D-FIRE-HOTKEYS">
document.addEventListener("keydown", function(e) {

    const active = document.activeElement;

    if (
        active &&
        ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)
    ) {
        return;
    }

    const keys = {
        "a": "approve",
        "r": "relabel",
        "q": "quarantine",
        "s": "skip"
    };

    const status = keys[e.key.toLowerCase()];

    if (!status) {
        return;
    }

    e.preventDefault();

    const button = document.querySelector(
        'button[value="' + status + '"]'
    );

    if (button) {
        button.click();
    }
});
</script>

</body>

</html>
"""


@app.route("/")
def index():

    rows = load_queue()

    idx = int(request.args.get("i", 0))
    idx = max(0, min(idx, len(rows) - 1))

    reviewed = sum(
        1
        for r in rows
        if r["status"] not in ("", "unreviewed")
    )

    return render_template_string(
        HTML,
        row=rows[idx],
        index=idx,
        total=len(rows),
        reviewed=reviewed,
        prev_index=max(0, idx - 1),
        next_index=min(len(rows) - 1, idx + 1),
    )


@app.route("/image/<int:index>")
def image(index):

    rows = load_queue()

    if index < 0 or index >= len(rows):
        return "Invalid index", 404

    row = rows[index]

    image = cv2.imread(row["image"])

    if image is None:
        return "Cannot read image", 500

    image = draw_boxes(
        image,
        row["label"]
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    if not ok:
        return "Encode error", 500

    return Response(
        encoded.tobytes(),
        mimetype="image/jpeg"
    )


@app.route("/mark/<int:index>", methods=["POST"])
def mark(index):

    rows = load_queue()

    if index < 0 or index >= len(rows):
        return "Invalid index", 404

    status = request.form.get(
        "status",
        "skip"
    )

    note = request.form.get(
        "note",
        ""
    ).strip()

    rows[index]["status"] = status
    rows[index]["note"] = note

    save_queue(rows)

    next_index = min(
        index + 1,
        len(rows) - 1
    )

    return redirect(
        f"/?i={next_index}"
    )


if __name__ == "__main__":

    build_queue()

    print("=" * 60)
    print("D-Fire Review")
    print("=" * 60)
    print(f"Queue : {QUEUE_FILE}")
    print("URL   : http://127.0.0.1:8765")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=8765,
        debug=False
    )
