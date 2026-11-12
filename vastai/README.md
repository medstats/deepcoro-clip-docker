# Vast.ai Deployment Guide: DeepCORO-CLIP

This guide provides instructions for launching and executing **DeepCORO-CLIP** on on-demand GPU instances via Vast.ai.

Container Image: `ghcr.io/medstats/deepcoro-clip-docker:latest`  
Vast.ai Template: **`DeepCORO-CLIP-GPU`**

---

## 1. Hardware Requirements

- **GPU:** Single NVIDIA GPU with $\ge 16\,\text{GB}$ VRAM (NVIDIA RTX 3090, RTX 4090, A5000, or A100 recommended).
- **Disk Space:** Minimum $30\,\text{GB}$ persistent disk.
- **Docker Flags:** `--ipc=host` recommended for PyTorch multi-frame video loading.

---

## 2. Launching from Vast.ai Web Console

1. In the Vast.ai Console, search for template **`DeepCORO-CLIP-GPU`** (or select custom image `ghcr.io/medstats/deepcoro-clip-docker:latest`).
2. Set disk allocation to **35 GB**.
3. Choose an RTX 3090 or RTX 4090 host ($\approx \$0.15 - \$0.25/\text{hr}$).
4. Launch the instance with Jupyter or SSH access.

---

## 3. Launching via Vast.ai CLI

```bash
# 1. Search for suitable RTX 3090 / 4090 offers
vastai search offers 'gpu_name in [RTX_3090, RTX_4090] num_gpus=1 inet_down > 200' -o 'dph'

# 2. Create the instance
vastai create instance <MACHINE_ID> \
    --image ghcr.io/medstats/deepcoro-clip-docker:latest \
    --disk 35 \
    --ssh \
    --args "bash /workspace/vastai/run.sh /workspace/data /workspace/results 0"
```

---

## 4. Running Inference Inside Instance

Once SSH'd or in terminal:

```bash
# Copy DICOM studies into /workspace/data
# Run the pipeline
python /workspace/run_deepcoro.py \
    --input /workspace/data \
    --output /workspace/results \
    --model stenosis \
    --gpu 0
```

---

## 5. Deliverables Generated

```text
/workspace/results/
├── summary.csv             # Review table with segment predictions & calibrated thresholds
├── predictions_raw.csv     # All 90 model raw predictions (continuous & sigmoid)
├── predictions.json        # Structured hierarchical JSON per patient study
├── run_metadata.json       # Audit trail (SHA256, commit, GPU, runtime)
└── inference.log           # Execution progress log
```
