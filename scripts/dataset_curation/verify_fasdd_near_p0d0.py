#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

IMAGES = DATA / "images"

CANDIDATES = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "candidate_pairs.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
)

OUT_CSV = OUT_DIR / "p0d0_pixel_metrics.csv"
OUT_JSON = OUT_DIR / "p0d0_pixel_profile.json"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

THUMB_SIZE = 128

GATES = {
    "ultra_strict": {
        "corr": 0.999,
        "mae": 0.010,
        "color": 0.995,
    },
    "very_strict": {
        "corr": 0.995,
        "mae": 0.020,
        "color": 0.990,
    },
    "strict": {
        "corr": 0.990,
        "mae": 0.030,
        "color": 0.980,
    },
}


def decoded_pixel_hash(image):
    h = hashlib.sha256()

    shape = np.asarray(
        image.shape,
        dtype=np.int32,
    )

    h.update(shape.tobytes())
    h.update(image.tobytes())

    return h.hexdigest()


def make_features(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot decode: {path}"
        )

    height, width = image.shape[:2]

    pixel_hash = decoded_pixel_hash(image)

    thumb = cv2.resize(
        image,
        (THUMB_SIZE, THUMB_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        thumb,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        thumb,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        hist,
        hist,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return {
        "width": width,
        "height": height,
        "pixel_hash": pixel_hash,
        "thumb": thumb,
        "gray": gray,
        "hist": hist,
    }


def gray_correlation(a, b):
    value = cv2.matchTemplate(
        a,
        b,
        cv2.TM_CCOEFF_NORMED,
    )[0, 0]

    if not np.isfinite(value):
        return 0.0

    return float(value)


def normalized_mae(a, b):
    x = a.astype(np.float32)
    y = b.astype(np.float32)

    return float(
        np.mean(
            np.abs(x - y)
        )
        / 255.0
    )


def color_correlation(a, b):
    value = cv2.compareHist(
        a,
        b,
        cv2.HISTCMP_CORREL,
    )

    if not math.isfinite(value):
        return 0.0

    return float(value)


def psnr(a, b):
    value = cv2.PSNR(a, b)

    if math.isinf(value):
        return 999.0

    return float(value)


def relation_is_cross(relation):
    return relation in {
        "train_val",
        "train_test",
        "val_test",
    }


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("FASDD pHash=0 + dHash=0 PIXEL VERIFIER")
    print("=" * 92)

    image_map = {
        p.stem: p
        for p in IMAGES.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        )
    }

    pairs = []
    stems = set()

    with CANDIDATES.open(
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            pd = int(row["phash_distance"])
            dd = int(row["dhash_distance"])

            if pd != 0 or dd != 0:
                continue

            pairs.append(row)

            stems.add(row["stem_a"])
            stems.add(row["stem_b"])

    print(
        "p0/d0 pairs :",
        f"{len(pairs):,}"
    )

    print(
        "Unique images:",
        f"{len(stems):,}"
    )

    features = {}
    failures = []

    cv2.setNumThreads(0)

    for i, stem in enumerate(
        sorted(stems),
        1,
    ):
        path = image_map.get(stem)

        if path is None:
            failures.append(stem)
            continue

        try:
            features[stem] = make_features(path)
        except Exception:
            failures.append(stem)

        if (
            i % 2000 == 0
            or i == len(stems)
        ):
            print(
                f"\rFeature extraction "
                f"{i}/{len(stems)} "
                f"({i/len(stems)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    output = []
    stats = Counter()
    relation_counts = Counter()
    gate_counts = defaultdict(Counter)

    metric_bins = {
        "corr": Counter(),
        "mae": Counter(),
        "psnr": Counter(),
        "color": Counter(),
    }

    for i, row in enumerate(
        pairs,
        1,
    ):
        stem_a = row["stem_a"]
        stem_b = row["stem_b"]

        if (
            stem_a not in features
            or stem_b not in features
        ):
            stats[
                "pair_feature_failure"
            ] += 1
            continue

        a = features[stem_a]
        b = features[stem_b]

        relation = row["relation"]
        cross = relation_is_cross(relation)

        exact_dims = (
            a["width"] == b["width"]
            and a["height"] == b["height"]
        )

        decoded_exact = (
            exact_dims
            and a["pixel_hash"]
            == b["pixel_hash"]
        )

        corr = gray_correlation(
            a["gray"],
            b["gray"],
        )

        mae = normalized_mae(
            a["gray"],
            b["gray"],
        )

        color = color_correlation(
            a["hist"],
            b["hist"],
        )

        pair_psnr = psnr(
            a["thumb"],
            b["thumb"],
        )

        stats["pairs"] += 1
        relation_counts[relation] += 1

        if cross:
            stats["cross_split"] += 1
        else:
            stats["same_split"] += 1

        if exact_dims:
            stats["exact_dimensions"] += 1

        if decoded_exact:
            stats[
                "decoded_pixel_exact"
            ] += 1

            if cross:
                stats[
                    "decoded_pixel_exact_cross"
                ] += 1

        passed_gates = []

        for gate_name, gate in GATES.items():
            passed = (
                corr >= gate["corr"]
                and mae <= gate["mae"]
                and color >= gate["color"]
            )

            if passed:
                passed_gates.append(
                    gate_name
                )

                gate_counts[
                    gate_name
                ]["all"] += 1

                gate_counts[
                    gate_name
                ][relation] += 1

                if cross:
                    gate_counts[
                        gate_name
                    ]["cross"] += 1

        output.append({
            "stem_a": stem_a,
            "split_a": row["split_a"],
            "stem_b": stem_b,
            "split_b": row["split_b"],
            "relation": relation,
            "exact_dimensions": int(exact_dims),
            "decoded_pixel_exact": int(decoded_exact),
            "gray_corr": f"{corr:.8f}",
            "gray_mae": f"{mae:.8f}",
            "psnr": f"{pair_psnr:.6f}",
            "color_corr": f"{color:.8f}",
            "passed_gates": "|".join(
                passed_gates
            ),
        })

        for threshold in (
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            0.9995,
        ):
            if corr >= threshold:
                metric_bins[
                    "corr"
                ][str(threshold)] += 1

        for threshold in (
            0.05,
            0.03,
            0.02,
            0.01,
            0.005,
            0.001,
        ):
            if mae <= threshold:
                metric_bins[
                    "mae"
                ][str(threshold)] += 1

        for threshold in (
            25,
            30,
            35,
            40,
            50,
        ):
            if pair_psnr >= threshold:
                metric_bins[
                    "psnr"
                ][str(threshold)] += 1

        for threshold in (
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
        ):
            if color >= threshold:
                metric_bins[
                    "color"
                ][str(threshold)] += 1

        if (
            i % 2000 == 0
            or i == len(pairs)
        ):
            print(
                f"\rVerifying "
                f"{i}/{len(pairs)} "
                f"({i/len(pairs)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    output.sort(
        key=lambda x: (
            0
            if x["decoded_pixel_exact"] == 1
            else 1,
            0
            if relation_is_cross(
                x["relation"]
            )
            else 1,
            -float(x["gray_corr"]),
            float(x["gray_mae"]),
        )
    )

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "stem_a",
            "split_a",
            "stem_b",
            "split_b",
            "relation",
            "exact_dimensions",
            "decoded_pixel_exact",
            "gray_corr",
            "gray_mae",
            "psnr",
            "color_corr",
            "passed_gates",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(output)

    report = {
        "candidate_filter":
            "phash_distance=0 AND dhash_distance=0",

        "pairs":
            len(pairs),

        "unique_images":
            len(stems),

        "feature_failures":
            failures,

        "stats":
            dict(stats),

        "relations":
            dict(relation_counts),

        "diagnostic_gates": {
            name: {
                "thresholds":
                    GATES[name],

                "counts":
                    dict(
                        gate_counts[name]
                    ),
            }
            for name in GATES
        },

        "metric_bins": {
            key:
                dict(value)
            for key, value
            in metric_bins.items()
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("p0/d0 PIXEL PROFILE")
    print("=" * 92)

    print(
        "Pairs                 :",
        f"{stats['pairs']:,}"
    )

    print(
        "Same split            :",
        f"{stats['same_split']:,}"
    )

    print(
        "Cross split           :",
        f"{stats['cross_split']:,}"
    )

    print(
        "Exact dimensions      :",
        f"{stats['exact_dimensions']:,}"
    )

    print(
        "Decoded-pixel exact   :",
        f"{stats['decoded_pixel_exact']:,}"
    )

    print(
        "Decoded exact cross   :",
        f"{stats['decoded_pixel_exact_cross']:,}"
    )

    print()
    print("DIAGNOSTIC GATES")
    print("-" * 92)

    for gate_name in GATES:
        c = gate_counts[gate_name]

        print()
        print(gate_name)

        print(
            "  all   :",
            f"{c['all']:,}"
        )

        print(
            "  cross :",
            f"{c['cross']:,}"
        )

        for rel in (
            "train_val",
            "train_test",
            "val_test",
        ):
            print(
                f"  {rel:12}: "
                f"{c[rel]:,}"
            )

    print()
    print("METRIC PROFILE")
    print("-" * 92)

    print()
    print("Gray correlation >=")

    for k, v in metric_bins[
        "corr"
    ].items():
        print(
            f"  {k:8}: {v:,}"
        )

    print()
    print("Gray MAE <=")

    for k, v in metric_bins[
        "mae"
    ].items():
        print(
            f"  {k:8}: {v:,}"
        )

    print()
    print("PSNR >=")

    for k, v in metric_bins[
        "psnr"
    ].items():
        print(
            f"  {k:8}: {v:,}"
        )

    print()
    print("Color correlation >=")

    for k, v in metric_bins[
        "color"
    ].items():
        print(
            f"  {k:8}: {v:,}"
        )

    print()
    print(
        "CSV :",
        OUT_CSV
    )

    print(
        "JSON:",
        OUT_JSON
    )

    print()
    print(
        "No dataset or manifest was modified."
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
