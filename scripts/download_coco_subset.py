"""
COCO Training Subset Builder
============================
Builds a class-balanced subset of COCO train2017 and downloads only the images
it needs, avoiding the 19 GB full-archive download.

    py -3.14 scripts/download_coco_subset.py --root ./coco --images 25000

The subset is selected from ``instances_train2017.json`` (which ships with the
annotations archive, so no image data is required to plan the download). A
greedy class-balancing pass prioritizes images containing under-represented
categories, because a uniformly random subset of COCO badly under-samples rare
classes such as hair drier and toaster, and mAP averages over all 80 classes
equally.

Selection is seeded, so the same subset is reproduced on every run and a
reported number can be tied to a specific set of images.

Downloads run in a thread pool and skip files already present, so an
interrupted run can simply be re-run.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def select_balanced_subset(data, n_images, seed=0):
    """Pick ``n_images`` image ids, prioritizing rare-category coverage.

    Images are scored by the rarity of the categories they contain, so each
    pick maximizes the gain for whichever classes are currently least covered.
    """
    rng = random.Random(seed)

    per_image_cats = defaultdict(set)
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            per_image_cats[ann["image_id"]].add(ann["category_id"])

    # Images with no annotations carry no learning signal for detection.
    candidates = [img_id for img_id, cats in per_image_cats.items() if cats]
    rng.shuffle(candidates)

    if n_images >= len(candidates):
        print(f"  requested {n_images} >= {len(candidates)} annotated images; using all")
        return set(candidates)

    cat_counts = defaultdict(int)
    target_per_cat = max(1, n_images * 3 // max(len(data["categories"]), 1))
    selected = set()

    # First pass: greedily cover rare categories.
    by_rarity = sorted(
        candidates,
        key=lambda i: min((cat_counts[c] for c in per_image_cats[i]), default=0),
    )
    for img_id in by_rarity:
        if len(selected) >= n_images:
            break
        cats = per_image_cats[img_id]
        if any(cat_counts[c] < target_per_cat for c in cats):
            selected.add(img_id)
            for c in cats:
                cat_counts[c] += 1

    # Second pass: fill the remainder at random.
    for img_id in candidates:
        if len(selected) >= n_images:
            break
        selected.add(img_id)

    return selected


def write_subset_annotations(data, selected_ids, out_path):
    """Write a COCO JSON containing only the selected images and their boxes."""
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [img for img in data["images"] if img["id"] in selected_ids],
        "annotations": [a for a in data["annotations"] if a["image_id"] in selected_ids],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f)
    return subset


def fetch_one(file_name, dest_dir, timeout=30, retries=3):
    """Download a single image unless it already exists. Returns bytes written."""
    dest = Path(dest_dir) / file_name
    if dest.exists() and dest.stat().st_size > 0:
        return 0

    url = IMAGE_URL.format(file_name=file_name)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return len(payload)
        except Exception:
            if attempt == retries - 1:
                return -1
            time.sleep(1.5 * (attempt + 1))
    return -1


def main():
    parser = argparse.ArgumentParser(description="Build and fetch a COCO train subset")
    parser.add_argument("--root", default="./coco")
    parser.add_argument("--images", type=int, default=25000,
                        help="subset size (COCO minitrain uses 25000)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true",
                        help="write the subset annotations without downloading images")
    args = parser.parse_args()

    root = Path(args.root)
    ann_file = root / "annotations" / "instances_train2017.json"
    img_dir = root / "images" / "train2017"
    out_ann = root / "annotations" / "instances_minitrain2017.json"

    if not ann_file.exists():
        print(f"ERROR: {ann_file} not found.\n"
              f"Fetch the annotations first:\n"
              f"  py -3.14 download_coco.py --annotations-only")
        return 1

    print(f"Reading {ann_file.name} ({human(ann_file.stat().st_size)})...")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data['images'])} images, {len(data['annotations'])} annotations, "
          f"{len(data['categories'])} categories")

    print(f"Selecting a class-balanced subset of {args.images} images (seed {args.seed})...")
    selected = select_balanced_subset(data, args.images, seed=args.seed)
    subset = write_subset_annotations(data, selected, out_ann)
    print(f"  selected {len(subset['images'])} images, "
          f"{len(subset['annotations'])} annotations -> {out_ann.name}")

    covered = len({a["category_id"] for a in subset["annotations"]})
    print(f"  category coverage: {covered}/{len(data['categories'])}")

    if args.plan_only:
        print("\n--plan-only set; skipping image download.")
        return 0

    img_dir.mkdir(parents=True, exist_ok=True)
    needed = [img["file_name"] for img in subset["images"]
              if not (img_dir / img["file_name"]).exists()]
    print(f"\n{len(needed)} images to download "
          f"({len(subset['images']) - len(needed)} already present)")
    if not needed:
        print("Nothing to do.")
        return 0

    print(f"Estimated size: ~{human(len(needed) * 160_000)} "
          f"(COCO images average ~160 KB)\n")

    downloaded = failed = 0
    total_bytes = 0
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, fn, img_dir): fn for fn in needed}
        for i, future in enumerate(as_completed(futures), 1):
            size = future.result()
            if size < 0:
                failed += 1
            else:
                downloaded += 1
                total_bytes += size

            if i % 100 == 0 or i == len(needed):
                elapsed = time.perf_counter() - start
                rate = i / max(elapsed, 1e-6)
                eta = (len(needed) - i) / max(rate, 1e-6)
                sys.stdout.write(
                    f"\r  {i}/{len(needed)}  ok {downloaded}  failed {failed}  "
                    f"{human(total_bytes)}  {rate:.0f} img/s  ETA {eta/60:.1f} min   ")
                sys.stdout.flush()

    print(f"\n\nDownloaded {downloaded} images ({human(total_bytes)}) in "
          f"{(time.perf_counter() - start)/60:.1f} min")
    if failed:
        print(f"WARNING: {failed} images failed. Re-run to retry just those.")

    print(f"\nTrain with:\n"
          f"  py -3.14 train.py --dataset coco-minitrain --data-root {root} "
          f"--backbone resnet34 --epochs 40\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
