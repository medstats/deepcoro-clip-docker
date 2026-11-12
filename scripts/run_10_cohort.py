#!/usr/bin/env python3
"""
Download and Benchmark 10 Additional Coronary Angiography Studies
Spanning the full spectrum of CAD severity:
  - Severe CAD (SYNTAX > 25)
  - Moderate CAD (SYNTAX 10 - 25)
  - Mild/Minimal CAD (SYNTAX 4 - 10)
  - Normal Coronaries (SYNTAX 0.0)
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
import urllib.request
from pathlib import Path
import pandas as pd

HF_BASE = "https://huggingface.co/datasets/MesserMMP/coronary-angiography-syntax"

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_vram_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_ram_mb = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if res.returncode == 0:
                    vram = float(res.stdout.strip().split("\n")[0])
                    if vram > self.peak_vram_mb:
                        self.peak_vram_mb = vram
            except Exception:
                pass

            try:
                ram = psutil.virtual_memory().used / (1024 * 1024)
                cpu = psutil.cpu_percent(interval=None)
                if ram > self.peak_ram_mb:
                    self.peak_ram_mb = ram
                if cpu > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu
            except Exception:
                pass

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def main():
    workspace = Path("/workspace")
    data_dir = workspace / "data_10"
    output_dir = workspace / "results_10"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" 1. Downloading 10 New Coronary Angiography Validation Studies")
    print("=" * 80)

    siemens = get_json(f"{HF_BASE}/raw/main/test_data/siemens_20.json")
    philips = get_json(f"{HF_BASE}/raw/main/test_data/philips_97.json")

    # Select 10 diverse cases across CAD severity spectrum
    selected_studies = [
        {"id": "CASE_01_SYNTAX36", "gt": "Severe CAD (SYNTAX 36.08)", "syntax": 36.08, "study": philips[8]},
        {"id": "CASE_02_SYNTAX27", "gt": "Severe CAD (SYNTAX 26.83)", "syntax": 26.83, "study": siemens[6]},
        {"id": "CASE_03_SYNTAX26", "gt": "Severe CAD (SYNTAX 25.50)", "syntax": 25.50, "study": siemens[4]},
        {"id": "CASE_04_SYNTAX24", "gt": "Moderate CAD (SYNTAX 23.50)", "syntax": 23.50, "study": siemens[8]},
        {"id": "CASE_05_SYNTAX19", "gt": "Moderate CAD (SYNTAX 18.50)", "syntax": 18.50, "study": siemens[2]},
        {"id": "CASE_06_SYNTAX15", "gt": "Moderate CAD (SYNTAX 14.50)", "syntax": 14.50, "study": siemens[7]},
        {"id": "CASE_07_SYNTAX10", "gt": "Mild CAD (SYNTAX 10.00)", "syntax": 10.00, "study": siemens[1]},
        {"id": "CASE_08_SYNTAX04", "gt": "Minimal CAD (SYNTAX 4.17)", "syntax": 4.17, "study": siemens[13]},
        {"id": "CASE_09_SYNTAX00", "gt": "Normal Coronaries (SYNTAX 0.0)", "syntax": 0.0, "study": [p for p in philips if p.get("mean_syntax", 1) == 0.0][2]},
        {"id": "CASE_10_SYNTAX00", "gt": "Normal Coronaries (SYNTAX 0.0)", "syntax": 0.0, "study": [p for p in philips if p.get("mean_syntax", 1) == 0.0][3]},
    ]

    total_cines = 0
    t_dl_start = time.time()

    for item in selected_studies:
        c_id = item["id"]
        study = item["study"]
        uid = study["study_uid"]
        videos = study["videos"]
        c_dir = data_dir / c_id
        c_dir.mkdir(parents=True, exist_ok=True)
        total_cines += len(videos)

        print(f"\n[{c_id}] {item['gt']} | {len(videos)} cines")
        for i, vid in enumerate(videos, 1):
            raw_path = vid["path"].replace("./", "")
            fname = Path(raw_path).name
            if not fname.endswith(".dcm"):
                fname = f"{fname}.dcm"
            dest = c_dir / fname
            if dest.exists() and dest.stat().st_size > 1000:
                continue
            dl_url = f"{HF_BASE}/resolve/main/test_data/{raw_path}"
            print(f"  [{i}/{len(videos)}] ⬇ {fname}...", end="", flush=True)
            req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:
                f_out.write(resp.read())
            print(f" ✓ ({dest.stat().st_size / (1024*1024):.1f} MB)")

    dl_time = time.time() - t_dl_start
    print(f"\nDownloaded {total_cines} cines across 10 cases in {dl_time:.1f} s.")

    print("\n" + "=" * 80)
    print(" 2. Executing DeepCORO-CLIP Master Inference on 10 Cases")
    print("=" * 80)

    monitor = ResourceMonitor(interval=0.25)
    monitor.start()

    inf_start = time.time()
    cmd = [
        sys.executable,
        str(workspace / "run_deepcoro.py"),
        "--input", str(data_dir),
        "--output", str(output_dir),
        "--model", "stenosis",
        "--gpu", "0"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    inf_elapsed = time.time() - inf_start
    monitor.stop()
    monitor.join()

    print("\n" + "=" * 80)
    print(" 3. Benchmarking Summary (10-Case Cohort)")
    print("=" * 80)
    print(f"Total Inference Pipeline Time: {inf_elapsed:.2f} s ({inf_elapsed/60:.2f} min)")
    print(f"Average Time per Case: {inf_elapsed / 10.0:.2f} s / case")
    print(f"Peak GPU VRAM: {monitor.peak_vram_mb:.1f} MB ({monitor.peak_vram_mb / 1024.0:.2f} GB)")
    print(f"Peak Host RAM: {monitor.peak_ram_mb:.1f} MB ({monitor.peak_ram_mb / 1024.0:.2f} GB)")
    print(f"Peak Host CPU: {monitor.peak_cpu_percent:.1f} %")

    # Save benchmark metrics
    report = {
        "cases_count": 10,
        "total_cines": total_cines,
        "total_inference_time_s": round(inf_elapsed, 2),
        "avg_time_per_case_s": round(inf_elapsed / 10.0, 2),
        "peak_vram_mb": round(monitor.peak_vram_mb, 1),
        "peak_vram_gb": round(monitor.peak_vram_mb / 1024.0, 2),
        "peak_ram_gb": round(monitor.peak_ram_mb / 1024.0, 2),
        "peak_cpu_percent": round(monitor.peak_cpu_percent, 1),
        "cohort": [{"id": item["id"], "syntax": item["syntax"], "gt": item["gt"]} for item in selected_studies]
    }
    with open(output_dir / "benchmark_10_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
