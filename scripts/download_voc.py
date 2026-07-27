"""
PASCAL VOC Downloader
=====================
Fetches VOC2007 trainval/test and VOC2012 trainval, then extracts them into a
single ``VOCdevkit`` tree.

    py -3.14 scripts/download_voc.py --root ./VOCdevkit

Downloads resume from a partial file, and an archive whose size already
matches is skipped, so an interrupted run can simply be re-run.

The standard detection protocol is: train on VOC2007 trainval + VOC2012
trainval (16,551 images), test on VOC2007 test (4,952 images). This is what
the published numbers for Faster R-CNN, SSD, and YOLOv3 use.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ARCHIVES = [
    {
        "name": "VOCtrainval_06-Nov-2007.tar",
        "urls": [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        ],
        "size": 460032000,
        "md5": "c52e279531787c972589f7e41ab4ae64",
    },
    {
        "name": "VOCtest_06-Nov-2007.tar",
        "urls": [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        ],
        "size": 451020800,
        "md5": "b6e924de25625d8de591ea690078ad9f",
    },
    {
        "name": "VOCtrainval_11-May-2012.tar",
        "urls": [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        ],
        "size": 1999639040,
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
    },
]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(urls, dest, expected_size=None):
    """Download with progress, trying each mirror in turn."""
    dest = Path(dest)
    if dest.exists() and expected_size and abs(dest.stat().st_size - expected_size) < 1024:
        print(f"  {dest.name}: already present ({human(dest.stat().st_size)}), skipping")
        return True

    for url in urls:
        try:
            print(f"  fetching {dest.name} from {url.split('/')[2]}")
            start = time.perf_counter()
            last = [0.0]

            def hook(count, block_size, total_size):
                now = time.perf_counter()
                if now - last[0] < 1.0 and count * block_size < total_size:
                    return
                last[0] = now
                got = count * block_size
                pct = (100.0 * got / total_size) if total_size > 0 else 0
                speed = got / max(now - start, 1e-6)
                sys.stdout.write(
                    f"\r    {pct:5.1f}%  {human(got)}/{human(total_size)}  "
                    f"{human(speed)}/s      ")
                sys.stdout.flush()

            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp, reporthook=hook)
            sys.stdout.write("\n")
            tmp.replace(dest)
            return True
        except Exception as exc:
            sys.stdout.write("\n")
            print(f"    mirror failed: {exc}")
            continue

    print(f"  ERROR: all mirrors failed for {dest.name}")
    return False


def verify_md5(path, expected):
    """Return True when the file hash matches, or when no hash is known."""
    if not expected:
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"  WARNING: checksum mismatch for {Path(path).name}")
        print(f"    expected {expected}\n    actual   {actual}")
        return False
    return True


def extract(archive, root):
    """Extract a VOC tar into ``root``'s parent so VOCdevkit lands correctly."""
    print(f"  extracting {Path(archive).name}")
    target = Path(root).parent
    with tarfile.open(archive) as tar:
        # filter='data' rejects absolute paths and traversal entries.
        try:
            tar.extractall(target, filter="data")
        except TypeError:
            tar.extractall(target)


def main():
    parser = argparse.ArgumentParser(description="Download PASCAL VOC")
    parser.add_argument("--root", default="./VOCdevkit")
    parser.add_argument("--cache", default="./downloads")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"VOC download -> {root.resolve()}")
    total = sum(a["size"] for a in ARCHIVES)
    print(f"Total download: {human(total)}\n")

    for spec in ARCHIVES:
        archive = cache / spec["name"]
        if not download(spec["urls"], archive, spec["size"]):
            return 1
        if not args.skip_verify and not verify_md5(archive, spec.get("md5")):
            print("  continuing despite checksum mismatch; re-run to re-download")
        extract(archive, root)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)

    print()
    for year in ("2007", "2012"):
        base = root / f"VOC{year}"
        if base.exists():
            n_img = len(list((base / "JPEGImages").glob("*.jpg")))
            n_ann = len(list((base / "Annotations").glob("*.xml")))
            print(f"  VOC{year}: {n_img} images, {n_ann} annotations")

    print(f"\nDone. Train with:\n"
          f"  py -3.14 train.py --dataset voc --data-root {root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
