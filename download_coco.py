"""
COCO 2017 Dataset Downloader
=============================
Downloads and extracts COCO 2017 data for reliable detection training.
By default this script downloads both train2017 and val2017 images,
plus train/val annotations.
"""

import os
import sys
import zipfile
import urllib.request
import argparse
from pathlib import Path

# URLs for COCO 2017
URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip' # 241MB
}

def download_file(url: str, dest_path: str):
    """Download file with progress bar."""
    if os.path.exists(dest_path):
        print(f"[-] {os.path.basename(dest_path)} already exists. Skipping download.")
        return

    print(f"[*] Downloading {url}...")
    
    def report_progress(block_num, calc_sz, total_sz):
        downloaded = block_num * calc_sz
        percent = min(100.0, float(downloaded) / float(total_sz) * 100.0)
        sys.stdout.write(f"\r    Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_sz / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\n[+] Download complete.")

def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"[*] Extracting {os.path.basename(zip_path)} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[+] Extraction complete.")

def setup_coco(target_dir="coco", val_only=False):
    """Setup directory structure and download COCO data."""
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/zips", exist_ok=True)
    
    print("="*50)
    print("  COCO 2017 Dataset Setup")
    print("="*50)

    if not val_only:
        train_zip = f"{target_dir}/zips/train2017.zip"
        download_file(URLS['train2017'], train_zip)
        if not os.path.exists(f"{target_dir}/images/train2017"):
            extract_zip(train_zip, f"{target_dir}/images")

    # Download validation images
    val_zip = f"{target_dir}/zips/val2017.zip"
    download_file(URLS['val2017'], val_zip)
    if not os.path.exists(f"{target_dir}/images/val2017"):
        extract_zip(val_zip, f"{target_dir}/images")
    elif not os.path.exists(f"{target_dir}/images/train2017"):
        print("[!] val-only mode enabled: train2017 was not downloaded.")
        print("[!] Training with --train-split train will fail integrity checks unless you pass --allow-synthetic.")

    # Download Annotations
    ann_zip = f"{target_dir}/zips/annotations.zip"
    download_file(URLS['annotations'], ann_zip)
    if not os.path.exists(f"{target_dir}/annotations"):
        extract_zip(ann_zip, target_dir)
        
    print("="*50)
    print("  Dataset Setup Complete!")
    print(f"  Target directory: {os.path.abspath(target_dir)}")
    if val_only:
        print("  For evaluation-only flows, use: python train.py --data coco --train-split val --epochs 2")
    else:
        print("  You can now run: python train.py --data coco --epochs 2")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO 2017 data")
    parser.add_argument('--target-dir', type=str, default='coco', help='Output dataset directory')
    parser.add_argument('--val-only', action='store_true', help='Download only val2017 images (not train2017)')
    args = parser.parse_args()

    setup_coco(target_dir=args.target_dir, val_only=args.val_only)
