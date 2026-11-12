# DeepCORO-CLIP: Coronary Angiography Deep Learning Suite

A production-grade, GPU-accelerated Docker container for automated **invasive X-ray coronary angiography interpretation**, integrating **DeepCORO-CLIP multiview attention foundation models** and **VasoVision preprocessing**.

* Container Registry: `ghcr.io/medstats/deepcoro-clip-docker:latest`
* Vast.ai Template: **`DeepCORO-CLIP-GPU`** (Template ID: `800873`)
* GitHub Repository: [`medstats/deepcoro-clip-docker`](https://github.com/medstats/deepcoro-clip-docker)
* Checkpoint: `zcb8cu0l_20260213-033432/best_model_epoch_11.pt` (`heartwise/deepcoro_clip_generic`)
* Checkpoint SHA256: `4eac7676bc24930466d84f04f813fcbc1b9e70b056232cad603f18c84d008267`

> [!CAUTION]
> **Clinical Disclaimer:** Research model only — not validated for primary clinical decision-making or quantitative coronary angiography (QCA).

---

## 1. Clinical & Model Architecture Overview

DeepCORO-CLIP processes all multi-angle cine runs of a patient's coronary angiography procedure together using a **Multiscale Vision Transformer (mViT)** backbone with **multiview cross-video attention pooling** (up to 10 diagnostic runs per study, 16 frames per cine at $224 \times 224$ resolution).

### 90 Multi-Target Outputs per Study
For each of the **18 anatomical coronary segments**, the model outputs 5 distinct findings ($18 \times 5 = 90$ total outputs):

1. **Continuous Stenosis Estimate:** Continuous regression score reflecting lumen narrowing.
2. **Significant Stenosis Probability:** Calibrated probability of $\ge 70\%$ stenosis ($\ge 50\%$ for Left Main).
3. **Chronic Total Occlusion (CTO) Probability:** Likelihood of complete coronary occlusion.
4. **Intracoronary Thrombus Probability:** Likelihood of visible intraluminal filling defect.
5. **Moderate / Severe Calcification Probability:** Likelihood of fluoroscopic vascular wall calcification.

### 18 Anatomical Coronary Segments
- **Right Coronary System (RCA):** Proximal RCA, Mid RCA, Distal RCA, Posterior Descending Artery (PDA), Posterolateral Branch.
- **Left Main & LAD:** Left Main, Proximal LAD, Mid LAD, Distal LAD, First Diagonal (D1), Second Diagonal (D2).
- **Left Circumflex (LCx):** Proximal LCx, Mid LCx, Distal LCx, First Marginal (OM1), Second Marginal (OM2), Posterior LV Branch (LVP).
- **Intermediate:** Ramus / Intermediate Branch (Bx).

---

## 2. Decision Thresholds & Interpretation Rules

Based on official calibration on 3,194 internal studies (yielding **81% mean sensitivity** and **73% mean specificity** across segments):

| Segment | Stenosis Threshold | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: |
| **Proximal RCA** | `0.108` | 0.84 | 0.83 |
| **Mid RCA** | `0.128` | 0.82 | 0.74 |
| **Distal RCA** | `0.033` | 0.80 | 0.69 |
| **Posterior Descending (PDA)** | `0.030` | 0.79 | 0.68 |
| **Posterolateral** | `0.021` | 0.78 | 0.67 |
| **Left Main** | `0.055` | 0.86 | 0.84 |
| **Proximal LAD** | `0.115` | 0.84 | 0.78 |
| **Mid LAD** | `0.175` | 0.81 | 0.72 |
| **Distal LAD** | `0.037` | 0.80 | 0.71 |
| **First Diagonal (D1)** | `0.131` | 0.82 | 0.74 |
| **Second Diagonal (D2)** | `0.057` | 0.79 | 0.70 |
| **Proximal LCx** | `0.083` | 0.83 | 0.77 |
| **Mid LCx** | `0.060` | 0.81 | 0.71 |
| **Distal LCx** | `0.011` | 0.77 | 0.65 |
| **First Marginal (OM1)** | `0.072` | 0.82 | 0.75 |
| **Second Marginal (OM2)** | `0.077` | 0.80 | 0.72 |
| **Ramus / Intermediate** | `0.030` | 0.79 | 0.70 |
| **Posterior LV Branch** | `0.001` | 0.75 | 0.62 |

> [!IMPORTANT]
> **Critical Thresholding Rules:**
> 1. **Do NOT use a probability cut-off of 0.5:** Uncalibrated 0.5 drops mean sensitivity from 81% down to **18%**.
> 2. **Do NOT apply 70% threshold to the continuous stenosis score:** Continuous stenosis scores are compressed toward lower numerical ranges (sensitivity drops to **8%** if thresholded at 70). Treat it as a continuous ranking/correlation score only.
> 3. **Rare Findings:** 13 rare output heads (e.g., thrombus in small distal branches) lack sufficient positive training cases and are explicitly flagged as `"threshold_unavailable"`.

---

## 3. How to Elicit Inference

### Master 1-Command Execution (`run_deepcoro.py`)
Run the entire end-to-end pipeline (DICOM discovery + anonymized grouping + video conversion + VasoVision filtering + DeepCORO inference + threshold mapping):

```bash
python /workspace/run_deepcoro.py \
    --input /workspace/dicom_studies \
    --output /workspace/results \
    --model stenosis \
    --gpu 0
```

### Accepted Input Types
- **Directory of DICOMs:** Automatically discovers all multi-frame `.dcm` files across nested directories and groups them by `StudyInstanceUID`.
- **Single DICOM Study:** Path to a single DICOM file or folder.
- **CSV Manifest:** CSV containing a `DICOMPath` or `FileName` column.

### Advanced Flags
- `--max-clips 10`: Set maximum diagnostic cines per study (default: 10).
- `--keep-avi`: Retain intermediate converted AVI videos in the output folder.
- `--skip-vasovision`: Bypass VasoVision quality/contrast filter.

---

## 4. Output Deliverables

The runner generates 5 structured deliverables in the target `--output` directory:

```text
results/
├── summary.csv             # Primary clinician review table
├── predictions_raw.csv     # Raw 90 continuous and sigmoid outputs per study
├── predictions.json        # Hierarchical JSON grouped by study and segment
├── run_metadata.json       # Audit trail (SHA256, commit, GPU, runtime)
└── inference.log           # Full pipeline execution log
```

### Review Table Schema (`summary.csv`)
| Column | Description | Example |
| :--- | :--- | :--- |
| `StudyID` | Anonymized Study Identifier | `STUDY_001` |
| `CoronarySegment` | Anatomical Segment Key | `prox_lad` |
| `SegmentName` | Full Anatomical Name | `Proximal LAD` |
| `SignificantStenosisProb` | Probability of $\ge 70\%$ stenosis | `0.8421` |
| `SignificantStenosisThreshold` | Calibrated decision cut-off | `0.1150` |
| `ThresholdPositive` | Flag (`prob > threshold`) | `True` |
| `ContinuousStenosisScore` | Uncalibrated continuous score | `54.20` |
| `CTOProb` | Chronic total occlusion probability | `0.0210` |
| `CTOThreshold` | Calibrated CTO cut-off | `0.0450` |
| `CTOPositive` | CTO flag | `False` |
| `ThrombusProb` | Intracoronary thrombus probability | `0.0120` |
| `ThrombusThreshold` | Calibrated thrombus cut-off | `threshold_unavailable` |
| `CalcificationProb` | Calcification probability | `0.1420` |
| `CalcificationThreshold` | Moderate/severe calcification cut-off | `0.0820` |
| `CalcificationPositive` | Calcification flag | `True` |

---

## 5. Cloud & Vast.ai Execution

### Vast.ai Template
- **Template Name:** `DeepCORO-CLIP-GPU`
- **Docker Image:** `ghcr.io/medstats/deepcoro-clip-docker:latest`
- **Recommended GPU:** NVIDIA RTX 3090, RTX 4090, or A5000 (24 GB VRAM)
- **Disk Allocation:** 35 GB

```bash
# Launch via Vast.ai CLI
vastai create instance <MACHINE_ID> \
    --image ghcr.io/medstats/deepcoro-clip-docker:latest \
    --disk 35 \
    --args "bash /workspace/vastai/run.sh /workspace/data /workspace/results 0"
```

---

## 6. Google Colab / Kaggle 1-Cell Run

```python
# 1. Clone repository
!git clone https://github.com/medstats/deepcoro-clip-docker.git /content/deepcoro
%cd /content/deepcoro

# 2. Install dependencies & Orion
!pip install -q pydicom pylibjpeg pylibjpeg-libjpeg opencv-python-headless timm transformers moviepy==1.0.2 fvcore timesformer_pytorch stam_pytorch
!git clone https://github.com/HeartWise-AI/Orion.git /content/deepcoro/Orion
!pip install -q -e /content/deepcoro/Orion

# 3. Download official weights
!mkdir -p /content/deepcoro/weights/deepcoro_clip_generic/zcb8cu0l_20260213-033432 /content/deepcoro/weights/VasoVision
!curl -L -s -o /content/deepcoro/weights/deepcoro_clip_generic/zcb8cu0l_20260213-033432/best_model_epoch_11.pt https://huggingface.co/heartwise/deepcoro_clip_generic/resolve/main/zcb8cu0l_20260213-033432/best_model_epoch_11.pt
!curl -L -s -o /content/deepcoro/weights/VasoVision/vaso_vision.pt https://huggingface.co/heartwise/VasoVision/resolve/main/vaso_vision.pt

# 4. Run master CLI
!python /content/deepcoro/run_deepcoro.py --input /content/my_dicoms --output /content/results --gpu 0
```

---

## 7. Citations & Acknowledgments

* **DeepCORO-CLIP & VasoVision:** HeartWise AI (`https://github.com/HeartWise-AI/DeepCORO_CLIP`).
* **Manuscript:** Avram et al., *Multiview Deep Learning for Comprehensive Coronary Angiography Interpretation*, arXiv:2603.17675.
* **Orion & HeartWise_StatPlots:** Preprocessing and statistical evaluation suites by HeartWise AI.
