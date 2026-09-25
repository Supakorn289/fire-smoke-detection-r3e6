#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import json
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    ROOT
    / "sources"
    / "fasdd"
    / "FASDD_CV.zip"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
)

REPORT_FILE = (
    REPORT_DIR
    / "archive_inspection.json"
)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def extension(name):

    return Path(name).suffix.lower()


def parent_path(name):

    return str(
        Path(name).parent
    )


def read_text(zf, name):

    with zf.open(name) as f:

        return f.read().decode(
            "utf-8",
            errors="replace",
        )


def analyze_yolo_text(text):

    text = text.strip()

    if not text:

        return {
            "valid": True,
            "empty": True,
            "classes": [],
            "boxes": 0,
        }

    classes = []
    boxes = 0

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            return {
                "valid": False,
                "empty": False,
                "classes": [],
                "boxes": boxes,
            }

        try:

            cls = int(parts[0])

            coords = list(
                map(
                    float,
                    parts[1:]
                )
            )

        except ValueError:

            return {
                "valid": False,
                "empty": False,
                "classes": [],
                "boxes": boxes,
            }

        if not all(
            -0.001 <= x <= 1.001
            for x in coords
        ):

            return {
                "valid": False,
                "empty": False,
                "classes": [],
                "boxes": boxes,
            }

        classes.append(cls)
        boxes += 1

    return {
        "valid": True,
        "empty": False,
        "classes": classes,
        "boxes": boxes,
    }


def xml_classes(text):

    try:

        root = ET.fromstring(text)

    except ET.ParseError:

        return []

    names = []

    for obj in root.findall(".//object"):

        node = obj.find("name")

        if (
            node is not None
            and node.text
        ):

            names.append(
                node.text
                .strip()
                .lower()
            )

    return names


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            ZIP_PATH
        )

    print("=" * 78)
    print("FASDD_CV AUTOMATIC ARCHIVE INSPECTOR")
    print("=" * 78)

    print(
        "ZIP:",
        ZIP_PATH.resolve()
    )

    print("=" * 78)

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        infos = [
            x
            for x in zf.infolist()
            if not x.is_dir()
        ]

        names = [
            x.filename
            for x in infos
        ]

        # ====================================================
        # EXTENSIONS
        # ====================================================

        ext_counter = Counter(
            extension(name)
            for name in names
        )

        print()
        print("FILE TYPES")
        print("-" * 78)

        for ext, count in sorted(
            ext_counter.items()
        ):

            print(
                f"{ext or '(none)':10} "
                f"{count:8}"
            )

        # ====================================================
        # IMAGE FILES
        # ====================================================

        images = [
            name
            for name in names
            if extension(name)
            in IMAGE_EXTS
        ]

        txt_files = [
            name
            for name in names
            if extension(name)
            == ".txt"
        ]

        xml_files = [
            name
            for name in names
            if extension(name)
            == ".xml"
        ]

        json_files = [
            name
            for name in names
            if extension(name)
            == ".json"
        ]

        print()
        print("TOTALS")
        print("-" * 78)

        print(
            "Images :",
            len(images)
        )

        print(
            "TXT    :",
            len(txt_files)
        )

        print(
            "XML    :",
            len(xml_files)
        )

        print(
            "JSON   :",
            len(json_files)
        )

        # ====================================================
        # TXT DIRECTORY DISCOVERY
        # ====================================================

        txt_parents = Counter(
            parent_path(name)
            for name in txt_files
        )

        print()
        print("TOP TXT DIRECTORIES")
        print("-" * 78)

        for path, count in (
            txt_parents.most_common(20)
        ):

            print(
                f"{count:8}  {path}"
            )

        # ====================================================
        # YOLO PATH DISCOVERY
        # ====================================================

        explicit_yolo = [
            name
            for name in txt_files
            if "yolo" in name.lower()
        ]

        print()
        print("YOLO PATH DISCOVERY")
        print("-" * 78)

        print(
            "TXT paths containing 'yolo':",
            len(explicit_yolo)
        )

        yolo_candidates = (
            explicit_yolo
            if explicit_yolo
            else txt_files
        )

        # sample up to 500
        sample = (
            yolo_candidates[:500]
        )

        valid_yolo = 0
        empty_yolo = 0
        invalid_txt = 0

        class_ids = Counter()

        for name in sample:

            info = zf.getinfo(name)

            # skip massive metadata txt
            if info.file_size > 1024 * 1024:
                continue

            text = read_text(
                zf,
                name,
            )

            result = analyze_yolo_text(
                text
            )

            if result["valid"]:

                valid_yolo += 1

                if result["empty"]:
                    empty_yolo += 1

                class_ids.update(
                    result["classes"]
                )

            else:

                invalid_txt += 1

        print(
            "Sample checked :",
            len(sample)
        )

        print(
            "YOLO-valid     :",
            valid_yolo
        )

        print(
            "YOLO-empty     :",
            empty_yolo
        )

        print(
            "Non-YOLO TXT   :",
            invalid_txt
        )

        print(
            "Class IDs seen :",
            dict(
                sorted(
                    class_ids.items()
                )
            )
        )

        # ====================================================
        # POSSIBLE SPLIT FILES
        # ====================================================

        split_files = []

        for name in txt_files:

            base = (
                Path(name)
                .name
                .lower()
            )

            if base in {
                "train.txt",
                "val.txt",
                "valid.txt",
                "test.txt",
            }:

                split_files.append(
                    name
                )

        print()
        print("SPLIT METADATA")
        print("-" * 78)

        if split_files:

            for name in split_files:

                info = zf.getinfo(name)

                print(
                    f"{name} "
                    f"({info.file_size} bytes)"
                )

        else:

            print(
                "No train.txt/val.txt/test.txt "
                "found by exact filename."
            )

        # ====================================================
        # COCO JSON
        # ====================================================

        coco_reports = []

        print()
        print("COCO ANNOTATIONS")
        print("-" * 78)

        for name in json_files:

            lower = name.lower()

            if "coco_cv" not in lower:
                continue

            try:

                data = json.loads(
                    read_text(
                        zf,
                        name,
                    )
                )

            except Exception as exc:

                print(
                    f"{name}: "
                    f"JSON ERROR: {exc}"
                )

                continue

            categories = (
                data.get(
                    "categories",
                    []
                )
            )

            images_count = len(
                data.get(
                    "images",
                    []
                )
            )

            annotations_count = len(
                data.get(
                    "annotations",
                    []
                )
            )

            category_view = [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                }
                for c in categories
            ]

            print()
            print(name)

            print(
                "  images      :",
                images_count
            )

            print(
                "  annotations :",
                annotations_count
            )

            print(
                "  categories  :",
                category_view
            )

            coco_reports.append({
                "path": name,
                "images":
                    images_count,
                "annotations":
                    annotations_count,
                "categories":
                    category_view,
            })

        # ====================================================
        # XML ↔ YOLO CLASS MAPPING
        # ====================================================

        print()
        print("VOC ↔ YOLO CLASS MAPPING")
        print("-" * 78)

        xml_by_stem = {
            Path(name).stem: name
            for name in xml_files
        }

        mapping_votes = defaultdict(
            Counter
        )

        matched_pairs = 0

        # work from YOLO-like files
        for txt_name in (
            yolo_candidates[:5000]
        ):

            stem = Path(
                txt_name
            ).stem

            xml_name = (
                xml_by_stem.get(
                    stem
                )
            )

            if not xml_name:
                continue

            txt_result = analyze_yolo_text(
                read_text(
                    zf,
                    txt_name,
                )
            )

            if (
                not txt_result["valid"]
                or txt_result["empty"]
            ):
                continue

            voc_names = xml_classes(
                read_text(
                    zf,
                    xml_name,
                )
            )

            yolo_classes = (
                txt_result[
                    "classes"
                ]
            )

            if not voc_names:
                continue

            # safest mapping:
            # image contains only one semantic class
            # and YOLO uses only one ID
            voc_unique = set(
                voc_names
            )

            yolo_unique = set(
                yolo_classes
            )

            if (
                len(voc_unique) == 1
                and
                len(yolo_unique) == 1
            ):

                semantic = next(
                    iter(
                        voc_unique
                    )
                )

                class_id = next(
                    iter(
                        yolo_unique
                    )
                )

                mapping_votes[
                    class_id
                ][semantic] += 1

                matched_pairs += 1

            if matched_pairs >= 500:
                break

        if not mapping_votes:

            print(
                "No safe automatic mapping "
                "could be derived."
            )

        else:

            for class_id in sorted(
                mapping_votes
            ):

                print(
                    f"class {class_id}: "
                    f"{dict(mapping_votes[class_id])}"
                )

        # ====================================================
        # IMAGE PREFIX DISTRIBUTION
        # ====================================================

        prefixes = Counter()

        for name in images:

            base = (
                Path(name)
                .stem
            )

            lower = (
                base.lower()
            )

            if lower.startswith(
                "bothfireandsmoke"
            ):
                prefix = "bothFireAndSmoke"

            elif lower.startswith(
                "fire"
            ):
                prefix = "fire"

            elif lower.startswith(
                "smoke"
            ):
                prefix = "smoke"

            elif (
                "neither" in lower
                or "nofire" in lower
            ):
                prefix = "negative"

            else:
                prefix = "other"

            prefixes[
                prefix
            ] += 1

        print()
        print("IMAGE NAME CATEGORIES")
        print("-" * 78)

        for key, value in (
            prefixes.most_common()
        ):

            print(
                f"{key:20}: {value}"
            )

        # ====================================================
        # REPORT
        # ====================================================

        report = {
            "archive":
                str(
                    ZIP_PATH.resolve()
                ),

            "files":
                len(infos),

            "extensions":
                dict(
                    ext_counter
                ),

            "images":
                len(images),

            "txt":
                len(txt_files),

            "xml":
                len(xml_files),

            "json":
                len(json_files),

            "txt_directories":
                dict(
                    txt_parents
                ),

            "explicit_yolo_txt":
                len(explicit_yolo),

            "sample_yolo": {
                "sample":
                    len(sample),

                "valid":
                    valid_yolo,

                "empty":
                    empty_yolo,

                "invalid":
                    invalid_txt,

                "class_ids":
                    dict(
                        class_ids
                    ),
            },

            "split_files":
                split_files,

            "coco":
                coco_reports,

            "mapping_votes": {
                str(class_id):
                    dict(votes)

                for (
                    class_id,
                    votes
                )
                in mapping_votes.items()
            },

            "filename_categories":
                dict(
                    prefixes
                ),
        }

        REPORT_FILE.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 78)
    print("INSPECTION COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_FILE
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
