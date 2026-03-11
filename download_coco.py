"""
COCO 2017 Dataset Downloader
=============================
Downloads and extracts the COCO 2017 validation images (1GB) 
and train/val annotations (241MB) as a lightweight starting point 
for training with real data.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# URLs for COCO 2017
URLS = {
    # 'train2017': 'http://images.cocodataset.org/zips/train2017.zip',  # 18GB (Heavy!)
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',        # 1GB (Recommended for test)
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

def setup_coco(target_dir="coco"):
    """Setup directory structure and download COCO data."""
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/zips", exist_ok=True)
    
    print("="*50)
    print("  COCO 2017 Lightweight Dataset Setup")
    print("="*50)
    
    # Download Validation images
    val_zip = f"{target_dir}/zips/val2017.zip"
    download_file(URLS['val2017'], val_zip)
    if not os.path.exists(f"{target_dir}/images/val2017"):
        extract_zip(val_zip, f"{target_dir}/images")
        
    # Since we are using val2017 as a quick start, we will symlink/copy it to train2017 
    # so the train.py script doesn't fail looking for training data.
    # We just want to get real data flowing through the pipeline quickly.
    print("[*] Setting up validation set as training set for a quick start test...")
    if not os.path.exists(f"{target_dir}/images/train2017"):
        # Just rename/symlink logic depending on OS, for simplicity we create a junction or copy
        if sys.platform == "win32":
            import _winapi
            try:
                _winapi.CreateJunction(
                    os.path.abspath(f"{target_dir}/images/val2017"), 
                    os.path.abspath(f"{target_dir}/images/train2017")
                )
                print("    [+] Created junction: train2017 -> val2017")
            except Exception as e:
                print(f"    [-] Junction failed: {e}. You may need to copy the folder manually.")
        else:
            os.symlink("val2017", f"{target_dir}/images/train2017")
            print("    [+] Created symlink: train2017 -> val2017")

    # Download Annotations
    ann_zip = f"{target_dir}/zips/annotations.zip"
    download_file(URLS['annotations'], ann_zip)
    if not os.path.exists(f"{target_dir}/annotations"):
        extract_zip(ann_zip, target_dir)
        
    print("="*50)
    print("  Dataset Setup Complete!")
    print(f"  Target directory: {os.path.abspath(target_dir)}")
    print("  You can now run: python train.py --data coco --epochs 2")
    print("="*50)

if __name__ == "__main__":
    setup_coco()
