#!/usr/bin/env python3
"""
Download 4 authentic Coronary Angiography (DICOM XA) validation studies
from MesserMMP/coronary-angiography-syntax on Hugging Face:
  - 2 Positive Cases (Severe multi-vessel CAD, SYNTAX 45.17 and 29.67)
  - 2 Negative Cases (Completely normal coronary tree, SYNTAX 0.0)

Usage:
  python download_test_cohort.py /workspace/data
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/MesserMMP/coronary-angiography-syntax"

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_cohort(target_dir: str):
    base_dir = Path(target_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Downloading 4-Case Coronary Angiography Validation Cohort")
    print(" Source: MesserMMP/coronary-angiography-syntax (Hugging Face)")
    print("=" * 80)

    print("Fetching cohort index metadata...")
    siemens = get_json(f"{HF_BASE}/raw/main/test_data/siemens_20.json")
    philips = get_json(f"{HF_BASE}/raw/main/test_data/philips_97.json")

    # Select the 4 cases
    pos1 = siemens[18]  # SYNTAX 45.17
    pos2 = siemens[0]   # SYNTAX 29.67
    
    # Select 2 genuine SYNTAX 0.0 cases
    negatives = [p for p in philips if p.get("mean_syntax", 1) == 0.0]
    neg1 = negatives[0] # SYNTAX 0.0
    neg2 = negatives[1] # SYNTAX 0.0

    cohort = [
        {"case_id": "POS_001_SYNTAX45", "label": "Positive (Severe CAD, SYNTAX 45.17)", "study": pos1},
        {"case_id": "POS_002_SYNTAX30", "label": "Positive (Severe CAD, SYNTAX 29.67)", "study": pos2},
        {"case_id": "NEG_001_SYNTAX0", "label": "Negative (Normal Coronaries, SYNTAX 0.0)", "study": neg1},
        {"case_id": "NEG_002_SYNTAX0", "label": "Negative (Normal Coronaries, SYNTAX 0.0)", "study": neg2},
    ]

    for item in cohort:
        case_id = item["case_id"]
        label = item["label"]
        study = item["study"]
        study_uid = study["study_uid"]
        videos = study["videos"]

        case_dir = base_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{case_id}] Ground Truth: {label}")
        print(f"  StudyInstanceUID: {study_uid}")
        print(f"  Cine Runs: {len(videos)} runs")

        for i, vid in enumerate(videos, 1):
            raw_path = vid["path"].replace("./", "")
            # Determine destination filename
            fname = Path(raw_path).name
            if not fname.endswith(".dcm"):
                fname = f"{fname}.dcm"
            
            dest_path = case_dir / fname
            if dest_path.exists() and dest_path.stat().st_size > 1000:
                print(f"  [{i}/{len(videos)}] {fname} already downloaded ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
                continue

            dl_url = f"{HF_BASE}/resolve/main/test_data/{raw_path}"
            print(f"  [{i}/{len(videos)}] ⬇ Downloading {fname}...", end="", flush=True)
            
            try:
                req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f_out:
                    f_out.write(resp.read())
                sz_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f" ✓ ({sz_mb:.1f} MB)")
            except Exception as e:
                print(f" ✗ Error: {e}")
                raise

    print("\n" + "=" * 80)
    print(" Validation cohort download complete!")
    print(f" Output folder: {base_dir}")
    print("=" * 80)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data"
    download_cohort(out_dir)
