"""
Filters mislabeled tiles using the right model for each category:

  YOLO (yolo11n.pt, COCO 80):  bicycles, boats, buses, cars,
                                hydrants, lights, meters, motorcycles
  OIV7 (yolov8n-oiv7.pt, 601): stairs
  CLIP (ViT-B/32):              bridges, chimneys, crosswalks, hills
                                (scene-level — detection models can't handle these)

Usage:
    pip install transformers   # for CLIP
    python clean.py --restore
    python clean.py                                   # all categories, all models
    python clean.py --yolo bicycles cars              # COCO only, specific folders
    python clean.py --oiv7 stairs                     # OIV7 only
    python clean.py --clip bridges chimneys           # CLIP only, specific folders
    python clean.py --dry-run                         # preview counts, no moves
"""

import argparse
import gc
import shutil
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

BASE_FOLDER     = Path("images")
REJECTED_FOLDER = Path("rejected")

CATEGORY_TO_COCO: dict[str, list[str]] = {
    "bicycles":    ["bicycle"],
    "boats":       ["boat"],
    "buses":       ["bus"],
    "cars":        ["car"],
    "hydrants":    ["fire hydrant"],
    "lights":      ["traffic light"],
    "meters":      ["parking meter"],
    "motorcycles": ["motorcycle"],
}

CATEGORY_TO_OIV7: dict[str, int] = {
    "stairs": 489,   # "Stairs" in Open Images V7
}

# Positive text prompts — CLIP keeps tiles that match ANY of these
CATEGORY_TO_CLIP: dict[str, list[str]] = {
    "bridges":    [
        "a bridge over water",
        "a road bridge or overpass",
        "a pedestrian bridge",
        "a stone or concrete bridge",
    ],
    "chimneys":   [
        "a chimney on a rooftop",
        "a brick chimney",
        "an industrial smokestack or chimney",
    ],
    "crosswalks": [
        "a crosswalk with white stripes on the road",
        "a pedestrian crossing on a street",
        "zebra crossing white lines on pavement",
    ],
    "hills":      [
        "a grassy hill or hillside",
        "rolling hills in a landscape",
        "a mountain or steep hill",
        "a hill with grass or trees",
    ],
}

def restore_all():
    if not REJECTED_FOLDER.exists():
        print("Nothing to restore — rejected/ folder not found.")
        return
    total = 0
    for cat_dir in sorted(REJECTED_FOLDER.iterdir()):
        if not cat_dir.is_dir():
            continue
        dest = BASE_FOLDER / cat_dir.name
        dest.mkdir(exist_ok=True)
        files = list(cat_dir.glob("*.png"))
        for f in files:
            shutil.move(str(f), dest / f.name)
        print(f"  Restored {len(files):>5} → images/{cat_dir.name}/")
        total += len(files)
    print(f"\nTotal restored: {total}")

def _rejected_dir(category: str, dry_run: bool) -> Path:
    d = REJECTED_FOLDER / category
    if not dry_run:
        d.mkdir(parents=True, exist_ok=True)
    return d

def _flush(i: int):
    if i % 500 == 0 and i > 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _report(category: str, kept: int, rejected: int, dry_run: bool):
    action = "would reject" if dry_run else "rejected"
    print(f"  {category}: kept {kept}, {action} {rejected} / {kept + rejected} total")

def _coco_ids(model: YOLO, class_names: list[str]) -> set[int]:
    lut = {v.lower(): k for k, v in model.names.items()}
    return {lut[n.lower()] for n in class_names if n.lower() in lut}

def clean_coco(model: YOLO, category: str, conf: float, dry_run: bool):
    folder = BASE_FOLDER / category
    if not folder.exists():
        print(f"  Folder not found: {folder}"); return

    target_ids = _coco_ids(model, CATEGORY_TO_COCO[category])
    rd = _rejected_dir(category, dry_run)
    images = list(folder.glob("*.png"))
    kept = rejected = 0
    for i, p in enumerate(images):
        results = model(str(p), conf=conf, verbose=False)
        detected = {int(b.cls) for r in results for b in r.boxes}
        del results
        if target_ids & detected:
            kept += 1
        else:
            rejected += 1
            if not dry_run:
                p.rename(rd / p.name)
        _flush(i)
    _report(category, kept, rejected, dry_run)

def clean_oiv7(model: YOLO, category: str, conf: float, dry_run: bool):
    folder = BASE_FOLDER / category
    if not folder.exists():
        print(f"  Folder not found: {folder}"); return

    target_id = CATEGORY_TO_OIV7[category]
    rd = _rejected_dir(category, dry_run)
    images = list(folder.glob("*.png"))
    kept = rejected = 0
    for i, p in enumerate(images):
        results = model(str(p), conf=conf, verbose=False)
        detected = {int(b.cls) for r in results for b in r.boxes}
        del results
        if target_id in detected:
            kept += 1
        else:
            rejected += 1
            if not dry_run:
                p.rename(rd / p.name)
        _flush(i)
    _report(category, kept, rejected, dry_run)

def load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError("Run: pip install transformers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device

def clean_clip(clip_model, processor, device,
               category: str, threshold: float, batch_size: int, dry_run: bool):
    folder = BASE_FOLDER / category
    if not folder.exists():
        print(f"  Folder not found: {folder}"); return

    prompts = CATEGORY_TO_CLIP[category]
    negative = "a random street scene with none of the above"
    all_prompts = prompts + [negative]

    import torch.nn.functional as F

    # Encode text once
    with torch.no_grad():
        text_inputs = processor(text=all_prompts, return_tensors="pt",
                                padding=True, truncation=True).to(device)
        text_out   = clip_model.text_model(**text_inputs)
        text_feats = clip_model.text_projection(text_out.pooler_output)
        text_feats = F.normalize(text_feats, dim=-1)

    rd = _rejected_dir(category, dry_run)
    images = list(folder.glob("*.png"))
    kept = rejected = 0

    for batch_start in range(0, len(images), batch_size):
        batch_paths = images[batch_start: batch_start + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        with torch.no_grad():
            img_inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            img_out   = clip_model.vision_model(**img_inputs)
            img_feats = clip_model.visual_projection(img_out.pooler_output)
            img_feats = F.normalize(img_feats, dim=-1)
            sims = (img_feats @ text_feats.T)  # [batch, n_prompts]

        for j, p in enumerate(batch_paths):
            best_positive = sims[j, :-1].max().item()   # best among positive prompts
            if best_positive >= threshold:
                kept += 1
            else:
                rejected += 1
                if not dry_run:
                    p.rename(rd / p.name)

        for img in pil_imgs:
            img.close()
        del img_inputs, img_feats, sims
        _flush(batch_start)

    _report(category, kept, rejected, dry_run)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py --restore
  python clean.py                                  # all categories, all models
  python clean.py --yolo bicycles cars motorcycles
  python clean.py --oiv7 stairs
  python clean.py --clip bridges crosswalks chimneys hills
  python clean.py --yolo cars --clip bridges --dry-run
        """
    )
    parser.add_argument("--restore",     action="store_true")
    parser.add_argument("--yolo",        nargs="+", metavar="CAT",
                        help=f"COCO categories. Defaults: {' '.join(sorted(CATEGORY_TO_COCO))}")
    parser.add_argument("--oiv7",        nargs="+", metavar="CAT",
                        help=f"OIV7 categories. Defaults: {' '.join(sorted(CATEGORY_TO_OIV7))}")
    parser.add_argument("--clip",        nargs="+", metavar="CAT",
                        help=f"CLIP categories. Defaults: {' '.join(sorted(CATEGORY_TO_CLIP))}")
    parser.add_argument("--conf",        type=float, default=0.20,
                        help="YOLO/OIV7 confidence threshold (default 0.20)")
    parser.add_argument("--clip-thresh", type=float, default=0.28,
                        help="CLIP cosine similarity threshold (default 0.28)")
    parser.add_argument("--clip-batch",  type=int,   default=32,
                        help="CLIP batch size (default 32)")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore_all()
        return

    explicit   = any(x is not None for x in [args.yolo, args.oiv7, args.clip])
    yolo_cats  = args.yolo if args.yolo is not None else (sorted(CATEGORY_TO_COCO) if not explicit else [])
    oiv7_cats  = args.oiv7 if args.oiv7 is not None else (sorted(CATEGORY_TO_OIV7) if not explicit else [])
    clip_cats  = args.clip if args.clip is not None else (sorted(CATEGORY_TO_CLIP) if not explicit else [])

    print(f"conf={args.conf}  clip_thresh={args.clip_thresh}  dry_run={args.dry_run}\n")

    if yolo_cats:
        print("── YOLO COCO (yolo11n.pt) ──────────────")
        model = YOLO("yolo11n.pt")
        for cat in yolo_cats:
            print(f"[{cat}]")
            if cat not in CATEGORY_TO_COCO:
                print(f"  not in CATEGORY_TO_COCO — skipping"); continue
            clean_coco(model, cat, args.conf, args.dry_run)
        del model; gc.collect()

    if oiv7_cats:
        print("\n── YOLO OIV7 (yolov8n-oiv7.pt) ────────")
        model = YOLO("yolov8n-oiv7.pt")
        for cat in oiv7_cats:
            print(f"[{cat}]")
            if cat not in CATEGORY_TO_OIV7:
                print(f"  not in CATEGORY_TO_OIV7 — skipping"); continue
            clean_oiv7(model, cat, args.conf, args.dry_run)
        del model; gc.collect()

    if clip_cats:
        print("\n── CLIP (ViT-B/32) ─────────────────────")
        clip_model, processor, device = load_clip()
        for cat in clip_cats:
            print(f"[{cat}]")
            if cat not in CATEGORY_TO_CLIP:
                print(f"  not in CATEGORY_TO_CLIP — skipping"); continue
            clean_clip(clip_model, processor, device,
                       cat, args.clip_thresh, args.clip_batch, args.dry_run)
        del clip_model; gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
