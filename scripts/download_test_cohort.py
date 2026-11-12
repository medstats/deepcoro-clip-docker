#!/usr/bin/env python3
"""
Downloads a benchmark evaluation cohort of 4 real clinical coronary angiography studies:
- 2 Positive Studies (Severe Stenosis / Multi-Vessel CAD, SYNTAX > 28)
- 2 Negative Studies (Completely Normal Coronary Tree, SYNTAX = 0.0)

Source: Multicenter Coronary Angiography SYNTAX Dataset (MesserMMP / Open Source)
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

BASE_HF_URL = "https://huggingface.co/datasets/MesserMMP/coronary-angiography-syntax/resolve/main/test_data"

COHORT = {
    "POS_001_SYNTAX45": {
        "study_uid": "1.2.826.0.1.3680043.2.706.3980268831.34545.18589.176.150.199.088",
        "ground_truth": "Positive (Severe CAD, SYNTAX 45.17)",
        "subpath": "siemens_20",
        "cines": ["im00151", "im00274", "im00390", "im00488", "im00595"]
    },
    "POS_002_SYNTAX30": {
        "study_uid": "1.2.826.0.1.3680043.2.706.726371717.5903.19401.163.204.13.156182",
        "ground_truth": "Positive (Severe CAD, SYNTAX 29.67)",
        "subpath": "siemens_20",
        "cines": ["im00084", "im00171", "im00247", "im00306", "im00500"]
    },
    "NEG_001_SYNTAX0": {
        "study_uid": "1.3.46.670589.28.26690171363123020221222050003785195",
        "ground_truth": "Negative (Normal Coronary Tree, SYNTAX 0.0)",
        "subpath": "philips_97",
        "cines": ["IM-0004-0040.dcm", "IM-0002-0042.dcm", "IM-0006-0042.dcm", "IM-0005-0042.dcm", "IM-0007-0042.dcm"]
    },
    "NEG_002_SYNTAX0": {
        "study_uid": "1.3.46.670589.28.26690171363123020220513064022227970",
        "ground_truth": "Negative (Normal Coronary Tree, SYNTAX 0.0)",
        "subpath": "philips_97",
        "cines": ["IM-0004-0063.dcm", "IM-0002-0066.dcm", "IM-0006-0044.dcm", "IM-0005-0042.dcm", "IM-0007-0044.dcm"]
    }
}


def download_cohort(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    print("================================================================================")
    print(" Downloading 4-Case Coronary Angiography Validation Cohort")
    print("================================================================================")

    for case_id, info in COHORT.items():
        case_dir = target_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{case_id}] Ground Truth: {info['ground_truth']}")
        
        for cine_name in info["cines"]:
            file_url = f"{BASE_HF_URL}/{info['subpath']}/{info['study_uid']}/{cine_name}"
            # Ensure .dcm extension
            dest_name = cine_name if cine_name.endswith(".dcm") else f"{cine_name}.dcm"
            dest_path = case_dir / dest_name

            if dest_path.exists() and dest_path.stat().st_size > 100000:
                print(f"  ✓ {dest_name} (Cached, {dest_path.stat().st_size / (1024*1024):.1f} MB)")
            else:
                print(f"  ⬇ Downloading {dest_name}...")
                urllib.request.urlretrieve(file_url, str(dest_path))
                print(f"  ✓ Saved ({dest_path.stat().st_size / (1024*1024):.1f} MB)")

            manifest.append({
                "CaseID": case_id,
                "StudyInstanceUID": info["study_uid"],
                "GroundTruth": info["ground_truth"],
                "CineFile": dest_name,
                "DICOMPath": str(dest_path.resolve())
            })

    # Save manifest CSV
    manifest_csv = target_dir / "cohort_manifest.csv"
    import pandas as pd
    pd.DataFrame(manifest).to_csv(manifest_csv, index=False)
    print(f"\n[DONE] Cohort manifest saved to: {manifest_csv}")
    return manifest_csv


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/validation_cohort")
    download_cohort(out_dir)
