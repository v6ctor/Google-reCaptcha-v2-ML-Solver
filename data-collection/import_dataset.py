"""
Downloads the Google reCAPTCHA Image dataset from Dataset Ninja and copies
each image into the correct images/<category>/ folder.

The dataset uses Supervisely format. Each annotation JSON contains either:
  - objects[].classTitle  (bounding-box level label)
  - tags[].name           (image-level label / challenge category)

Images are already 120×120 — the same size as scraped reCAPTCHA tiles — so
they are copied as-is with no resizing.

Usage:
    pip install dataset-tools
    python import_dataset.py
    python import_dataset.py --dst ~/dataset-ninja   # custom download dir
    python import_dataset.py --dry-run               # preview counts only
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path

IMAGES_FOLDER = Path("images")

# Map every Supervisely class/tag name → our canonical folder name
CLASS_MAP: dict[str, str] = {
    "stair":         "stairs",
    "stairs":        "stairs",
    "crosswalk":     "crosswalks",
    "chimney":       "chimneys",
    "car":           "cars",
    "bus":           "buses",
    "bicycle":       "bicycles",
    "motorcycle":    "motorcycles",
    "hydrant":       "hydrants",
    "traffic_light": "lights",
    "bridge":        "bridges",
    # "palm" and "other" are intentionally omitted — not in our training set
}


def categories_for_annotation(ann_path: Path) -> set[str]:
    """Return the set of canonical category names for one annotation file."""
    try:
        data = json.loads(ann_path.read_text())
    except Exception:
        return set()

    found: set[str] = set()

    # 1. Object-level labels (bounding boxes)
    for obj in data.get("objects", []):
        title = obj.get("classTitle", "").lower().strip()
        if title in CLASS_MAP:
            found.add(CLASS_MAP[title])

    # 2. Image-level tags (challenge category tag)
    for tag in data.get("tags", []):
        name = tag.get("name", "").lower().strip()
        if name in CLASS_MAP:
            found.add(CLASS_MAP[name])

    return found


def iter_splits(dataset_root: Path):
    """Yield (img_path, ann_path) pairs for every split in the dataset."""
    for split_dir in sorted(dataset_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "meta.json":
            continue
        img_dir = split_dir / "img"
        ann_dir = split_dir / "ann"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            ann_path = ann_dir / (img_path.name + ".json") if ann_dir.exists() else None
            yield img_path, ann_path


def download_dataset(dst: Path):
    try:
        import dataset_tools as dtools
    except ImportError:
        raise ImportError(
            "dataset-tools is required.\n"
            "Install it with:  pip install dataset-tools"
        )
    print(f"Downloading Google Recaptcha Image dataset to {dst} ...")
    dtools.download(dataset="Google Recaptcha Image", dst_dir=str(dst))


def find_dataset_root(dst: Path) -> Path:
    """Locate the directory that contains meta.json after download."""
    for candidate in [
        dst / "Google Recaptcha Image",
        dst / "google-recaptcha-image",
        dst,
    ]:
        if (candidate / "meta.json").exists():
            return candidate
    # Fallback: walk one level deep
    for child in dst.iterdir():
        if child.is_dir() and (child / "meta.json").exists():
            return child
    raise FileNotFoundError(
        f"Could not find meta.json under {dst}. "
        "Check the download completed successfully."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=Path, default=Path.home() / "dataset-ninja",
                        help="Directory to download the dataset into")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if dataset is already on disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without copying files")
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.dst)

    dataset_root = find_dataset_root(args.dst)
    print(f"Dataset root: {dataset_root}\n")

    counters: dict[str, int] = {}
    skipped = 0

    for img_path, ann_path in iter_splits(dataset_root):
        categories = categories_for_annotation(ann_path) if ann_path and ann_path.exists() else set()

        if not categories:
            skipped += 1
            continue

        for cat in categories:
            dest_dir = IMAGES_FOLDER / cat
            if not args.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / f"ds_{uuid.uuid4().hex[:8]}{img_path.suffix}"
                shutil.copy2(img_path, dest_file)
            counters[cat] = counters.get(cat, 0) + 1

    print("── Import summary ────────────────────────")
    for cat, n in sorted(counters.items()):
        print(f"  {cat:<15} +{n}")
    print(f"  {'(skipped)':<15} {skipped}  (no label / unmapped class)")

    if args.dry_run:
        print("\nDry run — no files were copied.")
    else:
        total = sum(counters.values())
        print(f"\nCopied {total} images into images/")


if __name__ == "__main__":
    main()
