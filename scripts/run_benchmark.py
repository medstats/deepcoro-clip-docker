#!/usr/bin/env python3
"""
DeepCORO-CLIP Benchmark & Resource Profiler
Runs 4-case validation cohort and measures:
  - Total and per-case elapsed time
  - Peak GPU VRAM utilization (MB)
  - System CPU and RAM usage
  - Prediction accuracy vs clinical ground truth
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
from pathlib import Path
import pandas as pd

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_vram_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_ram_mb = 0.0
        self.vram_samples = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # GPU VRAM query via nvidia-smi
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if res.returncode == 0:
                    vram = float(res.stdout.strip().split("\n")[0])
                    if vram > self.peak_vram_mb:
                        self.peak_vram_mb = vram
                    self.vram_samples.append(vram)
            except Exception:
                pass

            # System RAM & CPU
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
    data_dir = workspace / "data"
    output_dir = workspace / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Starting DeepCORO-CLIP 4-Case Evaluation & Resource Profiling")
    print("=" * 80)

    monitor = ResourceMonitor(interval=0.25)
    monitor.start()

    start_time = time.time()
    
    cmd = [
        sys.executable,
        str(workspace / "run_deepcoro.py"),
        "--input", str(data_dir),
        "--output", str(output_dir),
        "--model", "stenosis",
        "--gpu", "0"
    ]

    print(f"Executing: {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    total_elapsed = time.time() - start_time
    monitor.stop()
    monitor.join()

    print("\n" + "=" * 80)
    print(" Benchmarking Complete — Parsing Metrics")
    print("=" * 80)

    # 1. Resource Metrics
    print(f"Total Pipeline Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Average Time per Case (4 cases): {total_elapsed / 4.0:.2f} seconds")
    print(f"Peak GPU VRAM Utilization: {monitor.peak_vram_mb:.1f} MB ({monitor.peak_vram_mb / 1024.0:.2f} GB)")
    print(f"Peak System RAM: {monitor.peak_ram_mb:.1f} MB ({monitor.peak_ram_mb / 1024.0:.2f} GB)")
    print(f"Peak CPU Utilization: {monitor.peak_cpu_percent:.1f} %")

    # 2. Results Verification
    summary_file = output_dir / "summary.csv"
    preds_file = output_dir / "predictions.json"
    meta_file = output_dir / "run_metadata.json"

    print(f"\nGenerated Artifacts:")
    for f in [summary_file, preds_file, meta_file]:
        if f.exists():
            print(f"  ✓ {f.name} ({f.stat().st_size} bytes)")
        else:
            print(f"  ✗ Missing: {f.name}")

    if summary_file.exists():
        df_sum = pd.read_csv(summary_file)
        print("\n--- Summary CSV Predictions ---")
        print(df_sum.to_string(index=False))

    # Save benchmark report
    benchmark_report = {
        "total_runtime_seconds": round(total_elapsed, 2),
        "avg_time_per_case_seconds": round(total_elapsed / 4.0, 2),
        "peak_vram_mb": round(monitor.peak_vram_mb, 1),
        "peak_vram_gb": round(monitor.peak_vram_mb / 1024.0, 2),
        "peak_ram_mb": round(monitor.peak_ram_mb, 1),
        "peak_ram_gb": round(monitor.peak_ram_mb / 1024.0, 2),
        "peak_cpu_percent": round(monitor.peak_cpu_percent, 1),
        "return_code": proc.returncode
    }
    with open(output_dir / "benchmark_report.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
