# DeepCORO-CLIP: Technical Gotchas, Clinical Traps, and Engineering Tricks

**Author / Maintainer:** Antigravity AI Engineering & Dr. Anupam Kumar Singh  
**Repository:** [`medstats/deepcoro-clip-docker`](https://github.com/medstats/deepcoro-clip-docker)  
**Target Architecture:** DeepCORO-CLIP (MViT-S 16-frame Multi-Video Attention Foundation Model + VasoVision Filter)  
**Date:** November 2026

---

## 1. Clinical Interpretation Gotchas

### 🚨 Gotcha 1: The Fatal 0.50 Probability Trap
* **The Trap:** Applying standard binary classification thresholding ($P \ge 0.50$) to DeepCORO-CLIP's sigmoid output probabilities.
* **Clinical Consequence:** Model sensitivity collapses from **81% down to 18%**. True severe multi-vessel CAD lesions are missed completely, leading to massive false-negative rates in critical territories like Proximal LAD and RCA.
* **Why this happens:** In uncalibrated multi-label video attention models trained on imbalanced clinical cohorts, the output logit distribution is shifted towards negative predictions.
* **The Solution / Trick:** **Always load and enforce segment-specific calibrated decision thresholds** (from `config/inference/zcb8cu0l_thresholds.json`). These cutoffs maximize the Youden index ($J = \text{Sensitivity} + \text{Specificity} - 1$):
  * **Left Main (`left_main`):** `0.0546` (Extremely sensitive to protect against missing fatal LM disease)
  * **Proximal LAD (`prox_lad`):** `0.1145`
  * **Mid LAD (`mid_lad`):** `0.1754`
  * **Proximal RCA (`prox_rca`):** `0.1079`
  * **Mid RCA (`mid_rca`):** `0.1284`
  * **Proximal LCx (`prox_lcx`):** `0.0830`
  * **First Marginal (`om1`):** `0.0723`
  * **Second Marginal (`om2`):** `0.0767`
  * **First Diagonal (`D1`):** `0.1308`
  * **Second Diagonal (`D2`):** `0.0568`

---

### 🚨 Gotcha 2: The Continuous Stenosis Compression Illusion
* **The Trap:** Thresholding the continuous regression score (`<seg>_stenosis_pred`) at $\ge 70\%$ to flag significant diameter stenosis.
* **Clinical Consequence:** Sensitivity drops to an unacceptable **8%**.
* **Why this happens:** The continuous regression head compresses predictions towards lower numerical values (e.g., a severe $80\%$ stenosis on QCA often outputs a continuous prediction score in the $30\% - 60\%$ range).
* **The Solution / Trick:**
  1. Use `SignificantStenosisProb` (with calibrated threshold) for the binary clinical call ("Is there a $\ge 70\%$ lesion?").
  2. Use `ContinuousStenosisScore` as a **relative severity index** or ranking score, never as a hard binary gate.

---

### 🚨 Gotcha 3: The "Threshold Unavailable" Fallback
* **The Trap:** Expecting valid decision thresholds for all 90 output heads.
* **Clinical Consequence:** For rare segments or secondary tasks (e.g., Distal LAD CTO, Ramus thrombus, Posterior LV calcification), the upstream HeartWise team did not have sufficient validated training cases to determine an empirical ROC threshold.
* **The Solution / Trick:** The master runner (`run_deepcoro.py`) explicitly flags these heads as `threshold_unavailable` and sets `ThresholdPositive: threshold_unavailable`. Do not treat these as negative or positive without clinician review.

---

## 2. Data Engineering & Pipeline Gotchas

### 🚨 Gotcha 4: The Greek Alpha Delimiter (`sep='α'`)
* **The Trap:** Saving intermediate manifest CSVs with standard commas (`,`) or tabs (`\t`).
* **Technical Consequence:** The internal dataset loaders (`orion` and `DeepCORO`) crash or misalign columns silently.
* **Why this happens:** Hospital DICOM series metadata and video lists often contain commas, quotes, and square brackets (e.g. `['/path/im1.avi', '/path/im2.avi']`). To prevent parsing bugs without escaping, HeartWise uses the Greek character **`α`** (`\u03b1`) as the CSV separator.
* **The Solution / Trick:** Always write and parse manifests using:
  ```python
  df.to_csv(manifest_path, sep='α', index=False)
  df = pd.read_csv(manifest_path, sep='α')
  ```

---

### 🚨 Gotcha 5: DICOM Photometric Interpretation & Inversion
* **The Trap:** Feeding raw uint16 DICOM pixel arrays directly into OpenCV or video encoders.
* **Technical Consequence:** Inverted contrast, clipped white frames, or reversed catheter visibility.
* **The Solution / Trick:** Check tag `(0028, 0004)` `PhotometricInterpretation`:
  * `MONOCHROME2`: Standard radiologic representation (contrast agent & bones are dark/black, background is bright).
  * `MONOCHROME1`: Inverted representation (air is white, contrast is black).
  * Normalize dynamic range to $0 - 255$ uint8, and apply inversion if `MONOCHROME1`:
    ```python
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        frame = 255 - frame
    ```

---

### 🚨 Gotcha 6: Backbone Temporal Constraint (Strictly 16 Frames)
* **The Trap:** Feeding variable-length video clips directly into the MViT-S encoder.
* **Technical Consequence:** PyTorch tensor dimension mismatch error during temporal positional embedding addition.
* **Why this happens:** The pre-trained MViT (Multiscale Vision Transformer) backbone enforces a fixed temporal dimension of $T = 16$.
* **The Solution / Trick:** Uniform temporal subsampling:
  * If cine has $>16$ frames (standard XA cines typically have 30–120 frames): uniformly sample exactly 16 indices:
    ```python
    indices = np.linspace(0, total_frames - 1, 16).astype(int)
    ```
  * If cine has $<16$ frames: loop/pad the final frame up to 16.

---

### 🚨 Gotcha 7: Study Pooling & The 10-Clip Cap (`'PAD'` Tokens)
* **The Trap:** Feeding all raw cines from an exhaustive diagnostic study (>20 cines) into the multi-video attention pool.
* **Technical Consequence:** Out-Of-Memory (OOM) on GPU, or silent truncation of diagnostic views.
* **The Solution / Trick:**
  * DeepCORO-CLIP accepts a maximum of **10 cines per study**.
  * Use **VasoVision** to filter non-diagnostic runs (pigtail catheter, left ventriculogram, test puffs, fluoroscopy without contrast) and keep the highest-quality Left and Right coronary injections.
  * If a study has $<10$ diagnostic cines (e.g., 6 or 7 runs), DeepCORO-CLIP pads the remaining slots with `'PAD'` tokens:
    ```python
    video_name: ['im1.avi', 'im2.avi', 'im3.avi', 'PAD', 'PAD', 'PAD', ...]
    ```
  * The attention mechanism masks out `'PAD'` positions automatically.

---

## 3. PyTorch & Model Runtime Gotchas

### 🚨 Gotcha 8: PyTorch Distributed Runner Requirement
* **The Trap:** Executing inference directly with `python scripts/main.py --base_config ...`.
* **Technical Consequence:** Script crashes with `RuntimeError: Default process group has not been initialized`.
* **Why this happens:** DeepCORO's training framework relies on PyTorch DDP (`torch.distributed`) primitives for data dispatch even when evaluating on a single GPU.
* **The Solution / Trick:** Always launch inference using `torch.distributed.run`:
  ```bash
  python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29500 \
    /workspace/scripts/main.py \
    --base_config /workspace/tmp/docker_runtime_inference.yaml
  ```

---

### 🚨 Gotcha 9: `run_mode: inference` vs `Split: test` (Missing Ground Truth)
* **The Trap:** Setting `Split: test` in the manifest without providing ground-truth annotation columns.
* **Technical Consequence:** PyTorch `KeyError: 'prox_lad_stenosis'` during batch collation.
* **The Solution / Trick:**
  1. In the input manifest CSV, set `Split = 'inference'` for all rows.
  2. In the runtime YAML configuration, set:
     ```yaml
     run_mode: "inference"
     ```
  This tells the dataset loader to bypass ground-truth label extraction and output predictions only.

---

### 🚨 Gotcha 10: VasoVision Dynamic PyTorchVideo CDN Drops
* **The Trap:** Running VasoVision in an isolated/firewalled environment without pre-cached PyTorchVideo weights.
* **Technical Consequence:** `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired>`.
* **Why this happens:** Orion's VasoVision script dynamically invokes `torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m')` which downloads `X3D_M.pyth` from Facebook's CDN at runtime.
* **The Solution / Trick:**
  1. Our master runner implements an **automatic resilient fallback**: if VasoVision fails due to network/SSL drops, it logs a warning and falls back to chronological/quality sorting so inference **never crashes**.
  2. For fully deterministic offline runs, stage `X3D_M.pyth` in `/root/.cache/torch/hub/checkpoints/X3D_M.pyth`.

---

### 🚨 Gotcha 11: Hugging Face Model Gating for CardioSyntax & MACE
* **The Trap:** Trying to run `--model syntax` without prior authorization from HeartWise.
* **Technical Consequence:** `HTTP 403: Your request to access model heartwise/deepcoro_clip_cardiosyntax is awaiting a review from the repo authors.`
* **Why this happens:** While `deepcoro_clip_generic` is public, `deepcoro_clip_cardiosyntax` and `deepcoro_clip_mace` are **gated models**.
* **The Solution / Trick:**
  * Request access on Hugging Face at [`heartwise/deepcoro_clip_cardiosyntax`](https://huggingface.co/heartwise/deepcoro_clip_cardiosyntax).
  * Set `HUGGINGFACE_API_KEY` in environment. Once approved, the weights (`best_model_epoch_19.pt`) can be pulled instantly.

---

## 4. Hardware & Resource Optimization Tricks

### 💡 Trick 1: Sizing VRAM for Multi-Video Inference
* The MViT backbone processing up to 10 video clips (each 16 frames of $512 \times 512$ resolution) requires **6.18 GB – 6.50 GB peak VRAM**.
* **Implication:** You do NOT need expensive A100/H100 enterprise cards ($2.00+/hr). DeepCORO-CLIP runs at peak performance on:
  * **RTX 3080 (10 GB / 12 GB)** (~$0.15/hr)
  * **RTX 3090 / 4090 (24 GB)** (~$0.20 – $0.35/hr)
  * **RTX A4000 / A5000 (16 GB / 24 GB)** (~$0.18/hr)

### 💡 Trick 2: Throughput & Speed Optimization
* In benchmark testing across 10 patient studies (89 cines):
  * **DICOM Parsing & AVI Export:** ~15.1 s total (~0.17 s / cine)
  * **MViT Multi-Video Forward Pass:** ~7.0 s total (**0.70 s / patient**)
* To achieve maximum throughput when processing hundreds of hospital studies:
  1. Mount input storage on fast NVMe SSD.
  2. Set `--num_workers 4` or `8` in runtime YAML to parallelize video decoding.
  3. Pre-convert DICOMs to AVI if running repeated ablation sweeps using `--keep-avi`.

---

## 5. Output Summary Quick Reference

| Output File | Key Content | Use Case |
| :--- | :--- | :--- |
| `summary.csv` | 18 rows per study (one per segment): Continuous score, significant stenosis probability, decision threshold, binary call, CTO, thrombus, calcification | Clinical reporting & physician review |
| `predictions_raw.csv` | All 90 sigmoid probabilities and continuous regression values across all heads | Algorithmic audit & research analysis |
| `predictions.json` | Hierarchical structured JSON with metadata and segment mappings | PACS / EHR API integration |
| `run_metadata.json` | Model SHA256 checksum, upstream commit, GPU assigned, elapsed execution time | Regulatory audit trail & provenance |
| `inference.log` | Complete line-by-line pipeline execution log | Debugging & performance profiling |
