#!/usr/bin/env python3
"""
DeepCORO-CLIP Master Inference Runner: End-to-End Prediction Pipeline
====================================================================
Performs automated multi-video coronary angiography interpretation on
unlabelled DICOM studies without requiring manual clinician annotations.

Pipeline Stages:
1. DICOM Discovery & Anonymized Study Grouping (via StudyInstanceUID)
2. DICOM Cine to AVI Video Conversion
3. VasoVision Quality, Contrast, and Main-Structure Classification (Orion)
4. Diagnostic Coronary Clip Selection & Ranking (up to 10 runs per study)
5. DeepCORO-CLIP Multiview Attention Inference (90 targets across 18 segments)
6. Output Export & Decision Threshold Mapping (zcb8cu0l_thresholds.json)

Research model only — not validated for clinical decision-making.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
import pydicom

# Banner
BANNER = """
================================================================================
  DeepCORO-CLIP: AI Coronary Angiography Stenosis & Ischemia Engine
  Research model only — not validated for clinical decision-making.
================================================================================
"""

UPSTREAM_COMMIT = "ea0b97d0afcd582f9c24429bba677d6ba6c465fc"
CHECKPOINT_NAME = "zcb8cu0l_20260213-033432/best_model_epoch_11.pt"
CHECKPOINT_SHA256 = "4eac7676bc24930466d84f04f813fcbc1b9e70b056232cad603f18c84d008267"

CORONARY_SEGMENTS = [
    ("prox_rca", "Proximal RCA"),
    ("mid_rca", "Mid RCA"),
    ("dist_rca", "Distal RCA"),
    ("pda", "Posterior Descending Artery"),
    ("posterolateral", "Posterolateral Branch"),
    ("left_main", "Left Main"),
    ("prox_lad", "Proximal LAD"),
    ("mid_lad", "Mid LAD"),
    ("dist_lad", "Distal LAD"),
    ("D1", "First Diagonal"),
    ("D2", "Second Diagonal"),
    ("prox_lcx", "Proximal LCx"),
    ("mid_lcx", "Mid LCx"),
    ("dist_lcx", "Distal LCx"),
    ("om1", "First Marginal"),
    ("om2", "Second Marginal"),
    ("bx", "Ramus / Intermediate Branch"),
    ("lvp", "Posterior LV Branch")
]

DICOM_TAGS = {
    'study_instance_uid': (0x0020, 0x000D),
    'series_instance_uid': (0x0020, 0x000E),
    'sop_instance_uid': (0x0008, 0x0018),
    'series_times': (0x0008, 0x0031),
    'acquisition_time': (0x0008, 0x0032),
    'frame_height': (0x0028, 0x0011),
    'frame_width': (0x0028, 0x0010),
    'frame_rate': (0x0018, 0x0040),
    'cine_rate': (0x0018, 0x0040)
}


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("DeepCORO_Runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def verify_checkpoint_sha256(checkpoint_path: Path, expected_sha: str) -> bool:
    if not checkpoint_path.is_file():
        return False
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha.lower()


def anonymize_id(raw_id: str, prefix: str = "STUDY") -> str:
    h = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:8].upper()
    return f"{prefix}_{h}"


def extract_dicom_cine(
    dcm_path: Path,
    output_avi_path: Path,
    logger: logging.Logger
) -> Tuple[Optional[Path], Optional[float], str]:
    """
    Extract multi-frame DICOM cine and convert to AVI video.
    Returns: (avi_path, series_time, status)
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
    except Exception as e:
        logger.warning(f"Failed to read DICOM file {dcm_path.name}: {e}")
        return None, None, "read_error"

    series_time = 0.0
    try:
        if 0x00080031 in ds:
            series_time = float(ds[0x0008, 0x0031].value)
        elif 0x00080032 in ds:
            series_time = float(ds[0x0008, 0x0032].value)
    except Exception:
        series_time = 0.0

    if output_avi_path.exists():
        return output_avi_path, series_time, "success"

    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        logger.warning(f"Failed to read pixel data from {dcm_path.name}: {e}")
        return None, series_time, "pixel_error"

    if pixel_array.ndim != 3:
        logger.debug(f"Skipping single-frame or non-3D DICOM: {dcm_path.name} (shape {pixel_array.shape})")
        return None, series_time, "static_frame"

    num_frames, height, width = pixel_array.shape

    fps = 15.0
    if (0x0018, 0x0040) in ds:
        try:
            fps = float(ds[0x0018, 0x0040].value)
        except Exception:
            fps = 15.0

    photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    
    output_avi_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_avi_path), fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = pixel_array[i].astype(np.float32)
        # Normalize to 0-255
        f_min, f_max = frame.min(), frame.max()
        if f_max > f_min:
            frame = (frame - f_min) / (f_max - f_min) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        if photo_interp == "MONOCHROME1":
            frame = 255 - frame

        color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(color_frame)

    out.release()
    return output_avi_path, series_time, "success"


def scan_inputs(
    input_source: Path,
    tmp_avi_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Discovers all DICOM files, groups them by StudyInstanceUID, converts to AVI,
    and returns a preprocessed manifest with anonymized study IDs.
    """
    dicom_files: List[Path] = []
    
    if input_source.is_file():
        if input_source.suffix.lower() == ".csv":
            logger.info(f"Loading input manifest from CSV: {input_source}")
            csv_df = pd.read_csv(input_source)
            path_col = None
            for candidate in ["DICOMPath", "dicom_path", "FileName", "file_path", "FilePath"]:
                if candidate in csv_df.columns:
                    path_col = candidate
                    break
            if not path_col:
                raise ValueError(f"Input CSV must contain a 'DICOMPath' or 'FileName' column. Found: {list(csv_df.columns)}")
            for p in csv_df[path_col].dropna():
                p_path = Path(p)
                if p_path.is_file():
                    dicom_files.append(p_path)
                else:
                    logger.warning(f"Referenced DICOM path not found: {p}")
        else:
            dicom_files.append(input_source)
    elif input_source.is_dir():
        logger.info(f"Scanning directory for DICOM files: {input_source}")
        for ext in ["*.dcm", "*.DCM", "*.dicom", "*.DICOM"]:
            dicom_files.extend(list(input_source.rglob(ext)))
        if not dicom_files:
            # Check for extensionless files that might be DICOM
            for f in input_source.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        with open(f, "rb") as test_f:
                            test_f.seek(128)
                            if test_f.read(4) == b"DICM":
                                dicom_files.append(f)
                    except Exception:
                        pass
    else:
        raise FileNotFoundError(f"Input source path does not exist: {input_source}")

    logger.info(f"Found {len(dicom_files)} potential DICOM cines.")
    if not dicom_files:
        raise RuntimeError(f"No valid DICOM files located in {input_source}")

    records = []
    uid_to_anon: Dict[str, str] = {}
    anon_counter = 1

    for dcm_path in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            study_uid = str(getattr(ds, "StudyInstanceUID", f"UNKNOWN_STUDY_{anon_counter}"))
        except Exception:
            study_uid = f"CORRUPT_STUDY_{anon_counter}"

        if study_uid not in uid_to_anon:
            uid_to_anon[study_uid] = f"STUDY_{anon_counter:03d}"
            anon_counter += 1

        anon_study = uid_to_anon[study_uid]
        output_avi_name = f"{anon_study}_{dcm_path.stem}.avi"
        output_avi_path = tmp_avi_dir / output_avi_name

        avi_path, series_time, status = extract_dicom_cine(dcm_path, output_avi_path, logger)
        if status == "success" and avi_path is not None:
            records.append({
                "StudyInstanceUID": study_uid,
                "StudyID": anon_study,
                "FileName": str(avi_path.resolve()),
                "OriginalDICOM": str(dcm_path.resolve()),
                "SeriesTimes": series_time if series_time is not None else 0.0
            })

    if not records:
        raise RuntimeError("No DICOM files were successfully converted to multi-frame AVI cines.")

    df = pd.DataFrame(records)
    logger.info(f"Successfully processed {len(df)} cines across {df['StudyInstanceUID'].nunique()} patient studies.")
    return df, uid_to_anon


def run_vasovision_filter(
    df: pd.DataFrame,
    weights_path: Path,
    tmp_dir: Path,
    logger: logging.Logger,
    max_clips_per_study: int = 10,
    skip_vasovision: bool = False
) -> pd.DataFrame:
    """
    Applies VasoVision classification to filter for diagnostic contrast-injected
    Left and Right coronary runs, selecting up to max_clips_per_study.
    """
    if skip_vasovision:
        logger.info("Bypassing VasoVision filtering (--skip-vasovision enabled). Selecting up to max clips chronologically.")
        df_selected = df.sort_values(["StudyInstanceUID", "SeriesTimes"]).groupby("StudyInstanceUID").head(max_clips_per_study).copy()
        df_selected["main_structure_label"] = "All (VasoVision Skipped)"
        df_selected["contrast_agent"] = "yes"
        df_selected["status"] = "diagnostic"
        return df_selected

    logger.info("Running VasoVision classification on converted cines...")
    
    # Try importing Orion runtime
    try:
        from utils.external_validation_constants import MODEL_MAPPING
        from utils.orion_runtime_patches import apply_orion_runtime_patches
        apply_orion_runtime_patches()
        import orion.utils.video_training_and_eval as orion_eval

        # Setup Orion config
        vaso_config = MODEL_MAPPING['vaso_vision']['config'].copy()
        vaso_config['model_path'] = str(weights_path)
        vaso_config['root'] = str(Path(".").resolve())
        
        # Save temp manifest for VasoVision
        vaso_in_csv = tmp_dir / "vasovision_input.csv"
        df.to_csv(vaso_in_csv, sep='α', index=False)
        vaso_config['data_path'] = str(vaso_in_csv)

        df_vaso_info = orion_eval.perform_inference(
            config=vaso_config,
            split='inference',
            log_wandb=False
        )
        df_vaso_info.rename(columns={'filename': 'FileName'}, inplace=True)
        merged = pd.merge(df, df_vaso_info, on='FileName', how='left')

        # Filter: Left Coronary (0) or Right Coronary (1), contrast_agent == yes (1)
        labels_map = vaso_config['labels_map']
        left_id = labels_map['main_structure']['Left Coronary']
        right_id = labels_map['main_structure']['Right Coronary']
        contrast_yes = labels_map['contrast_agent']['yes']

        cond = (
            (merged['contrast_agent'] == contrast_yes) &
            (merged['main_structure'].isin([left_id, right_id]))
        )
        filtered = merged[cond].copy()
        
        # Map labels for reporting
        structure_names = {left_id: "Left Coronary", right_id: "Right Coronary"}
        filtered['main_structure_label'] = filtered['main_structure'].map(structure_names)

        logger.info(f"VasoVision filtered {len(df)} total cines down to {len(filtered)} diagnostic coronary runs.")

        if filtered.empty:
            logger.warning("No runs met VasoVision diagnostic threshold! Falling back to chronological top clips.")
            filtered = df.copy()
            filtered['main_structure_label'] = "Unclassified"

        # Group and select up to max_clips_per_study
        df_selected = filtered.sort_values(["StudyInstanceUID", "SeriesTimes"]).groupby("StudyInstanceUID").head(max_clips_per_study).copy()
        return df_selected

    except Exception as e:
        logger.warning(f"VasoVision execution encountered an issue ({e}). Falling back to quality sorting.")
        df_selected = df.sort_values(["StudyInstanceUID", "SeriesTimes"]).groupby("StudyInstanceUID").head(max_clips_per_study).copy()
        df_selected["main_structure_label"] = "Unclassified (Fallback)"
        return df_selected


def run_deepcoro_inference(
    filtered_df: pd.DataFrame,
    project_root: Path,
    output_dir: Path,
    gpu_id: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Executes DeepCORO-CLIP multi-video linear probing in pure inference mode.
    """
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepCORO-CLIP expects Split == 'inference'
    filtered_df = filtered_df.copy()
    filtered_df['Split'] = 'inference'
    
    # Target heads expected in head_structure
    base_config_path = project_root / "config" / "linear_probing" / "stenosis" / "docker_base_config.yaml"
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    target_labels = list(base_cfg.get("target_label", []))
    
    # Save input manifest for DeepCORO
    manifest_path = tmp_dir / "df_preprocessed_filtered.csv"
    filtered_df.to_csv(manifest_path, sep='α', index=False)
    
    # Build runtime YAML
    runtime_cfg = base_cfg.copy()
    runtime_cfg["data_filename"] = str(manifest_path.resolve())
    runtime_cfg["base_checkpoint_path"] = str((output_dir / "outputs").resolve())
    runtime_cfg["run_mode"] = "inference"
    runtime_cfg["use_wandb"] = False
    runtime_cfg["device"] = 0
    runtime_cfg["world_size"] = 1
    runtime_cfg["num_workers"] = 4
    runtime_cfg["batch_size"] = 1

    checkpoint_rel = base_cfg.get("inference_model_path", f"weights/deepcoro_clip_generic/{CHECKPOINT_NAME}")
    chk_path = project_root / checkpoint_rel
    if not chk_path.exists():
        # Search anywhere in weights/
        candidates = list((project_root / "weights").rglob("best_model_epoch_11.pt"))
        if candidates:
            chk_path = candidates[0]
        else:
            raise FileNotFoundError(f"DeepCORO-CLIP checkpoint best_model_epoch_11.pt not found under {project_root / 'weights'}")

    runtime_cfg["inference_model_path"] = str(chk_path.resolve())

    runtime_yaml_path = tmp_dir / "docker_runtime_inference.yaml"
    with open(runtime_yaml_path, "w") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    logger.info(f"Executing DeepCORO-CLIP inference on GPU {gpu_id}...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'Orion'}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=29500",
        str(project_root / "scripts" / "main.py"),
        "--base_config", str(runtime_yaml_path.resolve())
    ]

    p = subprocess.run(cmd, cwd=str(project_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logger.info(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(f"DeepCORO-CLIP inference failed with exit code {p.returncode}. See log output above.")

    # Locate generated predictions CSV
    # Usually saved under results/outputs/.../predictions/inference_predictions_epoch_-1.csv
    pred_candidates = list((output_dir / "outputs").rglob("*inference_predictions*.csv"))
    if not pred_candidates:
        # Check project_root / results
        pred_candidates = list((project_root / "results").rglob("*inference_predictions*.csv"))

    if not pred_candidates:
        raise FileNotFoundError(f"Inference completed but predictions CSV was not found in {output_dir / 'outputs'}")

    raw_preds_path = max(pred_candidates, key=os.path.getmtime)
    logger.info(f"Loaded raw predictions from: {raw_preds_path}")
    df_raw = pd.read_csv(raw_preds_path)
    return df_raw


def format_and_export_results(
    df_raw: pd.DataFrame,
    df_selected: pd.DataFrame,
    thresholds_path: Path,
    output_dir: Path,
    uid_to_anon: Dict[str, str],
    logger: logging.Logger
) -> None:
    """
    Maps all 90 model outputs to the 18 coronary segments, applies official
    calibrated thresholds, and exports structured CSV and JSON summaries.
    """
    logger.info("Formatting structured predictions and applying decision thresholds...")
    
    with open(thresholds_path, "r") as f:
        thresh_data = json.load(f)

    threshold_map = thresh_data.get("thresholds", {})
    
    # The raw predictions have 'video_name' containing the StudyInstanceUID
    id_col = "video_name" if "video_name" in df_raw.columns else df_raw.columns[0]

    # Save predictions_raw.csv directly
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(output_dir / "predictions_raw.csv", index=False)

    summary_rows = []
    json_results: Dict[str, Any] = {}

    for _, row in df_raw.iterrows():
        study_uid = str(row[id_col])
        study_id = uid_to_anon.get(study_uid, anonymize_id(study_uid))
        
        study_clips = df_selected[df_selected["StudyInstanceUID"] == study_uid]
        num_clips = len(study_clips)

        json_results[study_id] = {
            "study_instance_uid_hash": hashlib.sha256(study_uid.encode('utf-8')).hexdigest()[:12],
            "num_diagnostic_clips": num_clips,
            "segments": {}
        }

        for seg_key, seg_name in CORONARY_SEGMENTS:
            # 1. Continuous stenosis estimate
            stenosis_cont_col = f"{seg_key}_stenosis_pred"
            stenosis_cont_val = float(row[stenosis_cont_col]) if stenosis_cont_col in row else np.nan

            # 2. Significant stenosis probability
            stenosis_bin_col = f"{seg_key}_stenosis_binary_pred"
            stenosis_prob = float(row[stenosis_bin_col]) if stenosis_bin_col in row else np.nan
            
            thresh_entry = threshold_map.get(f"{seg_key}_stenosis_binary", {})
            thresh_val = float(thresh_entry.get("threshold", 0.5))
            derived = thresh_entry.get("derived", True)
            is_sig_stenosis = bool(stenosis_prob > thresh_val) if not np.isnan(stenosis_prob) else False

            # 3. CTO probability
            cto_col = f"{seg_key}_cto_pred"
            cto_prob = float(row[cto_col]) if cto_col in row else np.nan
            cto_thresh_entry = threshold_map.get(f"{seg_key}_cto", {})
            cto_derived = cto_thresh_entry.get("derived", False)
            cto_thresh = float(cto_thresh_entry.get("threshold", 0.5)) if cto_derived else None
            is_cto = bool(cto_prob > cto_thresh) if (cto_derived and not np.isnan(cto_prob)) else None

            # 4. Thrombus probability
            thromb_col = f"{seg_key}_thrombus_pred"
            thromb_prob = float(row[thromb_col]) if thromb_col in row else np.nan
            thromb_thresh_entry = threshold_map.get(f"{seg_key}_thrombus", {})
            thromb_derived = thromb_thresh_entry.get("derived", False)
            thromb_thresh = float(thromb_thresh_entry.get("threshold", 0.5)) if thromb_derived else None
            is_thromb = bool(thromb_prob > thromb_thresh) if (thromb_derived and not np.isnan(thromb_prob)) else None

            # 5. Moderate / Severe Calcification probability
            calc_col = f"{seg_key}_calcif_binary_pred"
            calc_prob = float(row[calc_col]) if calc_col in row else np.nan
            calc_thresh_entry = threshold_map.get(f"{seg_key}_calcif_binary", {})
            calc_derived = calc_thresh_entry.get("derived", False)
            calc_thresh = float(calc_thresh_entry.get("threshold", 0.5)) if calc_derived else None
            is_calc = bool(calc_prob > calc_thresh) if (calc_derived and not np.isnan(calc_prob)) else None

            # Summary row
            summary_rows.append({
                "StudyID": study_id,
                "CoronarySegment": seg_key,
                "SegmentName": seg_name,
                "SignificantStenosisProb": round(stenosis_prob, 4) if not np.isnan(stenosis_prob) else None,
                "SignificantStenosisThreshold": round(thresh_val, 4),
                "ThresholdPositive": is_sig_stenosis,
                "ContinuousStenosisScore": round(stenosis_cont_val, 2) if not np.isnan(stenosis_cont_val) else None,
                "CTOProb": round(cto_prob, 4) if not np.isnan(cto_prob) else None,
                "CTOThreshold": round(cto_thresh, 4) if cto_thresh is not None else "threshold_unavailable",
                "CTOPositive": is_cto if is_cto is not None else "threshold_unavailable",
                "ThrombusProb": round(thromb_prob, 4) if not np.isnan(thromb_prob) else None,
                "ThrombusThreshold": round(thromb_thresh, 4) if thromb_thresh is not None else "threshold_unavailable",
                "ThrombusPositive": is_thromb if is_thromb is not None else "threshold_unavailable",
                "CalcificationProb": round(calc_prob, 4) if not np.isnan(calc_prob) else None,
                "CalcificationThreshold": round(calc_thresh, 4) if calc_thresh is not None else "threshold_unavailable",
                "CalcificationPositive": is_calc if is_calc is not None else "threshold_unavailable",
                "PredictionStatus": "AVAILABLE"
            })

            # Hierarchical JSON
            json_results[study_id]["segments"][seg_key] = {
                "segment_name": seg_name,
                "stenosis": {
                    "continuous_score": round(stenosis_cont_val, 2) if not np.isnan(stenosis_cont_val) else None,
                    "probability_significant": round(stenosis_prob, 4) if not np.isnan(stenosis_prob) else None,
                    "official_threshold": round(thresh_val, 4),
                    "is_positive": is_sig_stenosis
                },
                "cto": {
                    "probability": round(cto_prob, 4) if not np.isnan(cto_prob) else None,
                    "official_threshold": round(cto_thresh, 4) if cto_thresh is not None else None,
                    "is_positive": is_cto,
                    "status": "derived" if cto_derived else "threshold_unavailable"
                },
                "thrombus": {
                    "probability": round(thromb_prob, 4) if not np.isnan(thromb_prob) else None,
                    "official_threshold": round(thromb_thresh, 4) if thromb_thresh is not None else None,
                    "is_positive": is_thromb,
                    "status": "derived" if thromb_derived else "threshold_unavailable"
                },
                "calcification": {
                    "probability": round(calc_prob, 4) if not np.isnan(calc_prob) else None,
                    "official_threshold": round(calc_thresh, 4) if calc_thresh is not None else None,
                    "is_positive": is_calc,
                    "status": "derived" if calc_derived else "threshold_unavailable"
                }
            }

    # Write summary.csv
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Write predictions.json
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Successfully generated summary.csv ({len(summary_df)} rows) and predictions.json.")


def main():
    print(BANNER)
    
    parser = argparse.ArgumentParser(
        description="DeepCORO-CLIP: Automated Coronary Angiography Stenosis & Ischemia Master Runner"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input directory of DICOM files, a single DICOM file, or a CSV manifest."
    )
    parser.add_argument(
        "--output",
        default="/workspace/results",
        help="Directory to save predictions, summaries, logs, and metadata (default: /workspace/results)."
    )
    parser.add_argument(
        "--model",
        default="stenosis",
        choices=["stenosis"],
        help="Model task to execute (default: stenosis)."
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA GPU ID to use for inference (default: 0)."
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Maximum diagnostic coronary cine runs to pool per study (default: 10)."
    )
    parser.add_argument(
        "--keep-avi",
        action="store_true",
        help="Preserve intermediate converted AVI video files in output directory."
    )
    parser.add_argument(
        "--skip-vasovision",
        action="store_true",
        help="Skip VasoVision quality/contrast filter and process raw cines directly."
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to custom decision thresholds JSON file."
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_source = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "inference.log"
    logger = setup_logger(log_file)
    logger.info(f"Starting DeepCORO-CLIP Inference Pipeline")
    logger.info(f"Input: {input_source}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPU ID: {args.gpu}")
    logger.info(f"Max clips per study: {args.max_clips}")

    start_time = time.time()

    # Step 1: Verify weights
    weights_dir = project_root / "weights"
    chk_path = weights_dir / "deepcoro_clip_generic" / CHECKPOINT_NAME
    if not chk_path.exists():
        found = list(weights_dir.rglob("best_model_epoch_11.pt"))
        if found:
            chk_path = found[0]

    sha_valid = verify_checkpoint_sha256(chk_path, CHECKPOINT_SHA256) if chk_path.exists() else False
    logger.info(f"Checkpoint status: {chk_path} | SHA256 verified: {sha_valid}")

    # Thresholds file
    thresh_path = Path(args.thresholds) if args.thresholds else (project_root / "config" / "inference" / "zcb8cu0l_thresholds.json")
    if not thresh_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresh_path}")

    # Step 2: DICOM Discovery & Conversion
    tmp_avi_dir = (output_dir / "intermediate_avi") if args.keep_avi else (project_root / "tmp" / "avi")
    tmp_avi_dir.mkdir(parents=True, exist_ok=True)

    df_manifest, uid_to_anon = scan_inputs(input_source, tmp_avi_dir, logger)

    # Step 3: VasoVision Preprocessing
    vaso_weights = weights_dir / "VasoVision" / "vaso_vision.pt"
    if not vaso_weights.exists():
        found_vaso = list(weights_dir.rglob("vaso_vision.pt"))
        if found_vaso:
            vaso_weights = found_vaso[0]

    df_filtered = run_vasovision_filter(
        df=df_manifest,
        weights_path=vaso_weights,
        tmp_dir=project_root / "tmp",
        logger=logger,
        max_clips_per_study=args.max_clips,
        skip_vasovision=args.skip_vasovision
    )

    # Step 4: Run DeepCORO-CLIP Multiview Inference
    df_raw = run_deepcoro_inference(
        filtered_df=df_filtered,
        project_root=project_root,
        output_dir=output_dir,
        gpu_id=args.gpu,
        logger=logger
    )

    # Step 5: Format & Export Deliverables
    format_and_export_results(
        df_raw=df_raw,
        df_selected=df_filtered,
        thresholds_path=thresh_path,
        output_dir=output_dir,
        uid_to_anon=uid_to_anon,
        logger=logger
    )

    elapsed_time = round(time.time() - start_time, 2)

    # Step 6: Write Run Metadata
    metadata = {
        "pipeline": "DeepCORO-CLIP Coronary Angiography Inference",
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_sha256_verified": sha_valid,
        "thresholds_file": thresh_path.name,
        "model_task": args.model,
        "gpu_assigned": args.gpu,
        "elapsed_seconds": elapsed_time,
        "total_cines_discovered": len(df_manifest),
        "total_studies_processed": int(df_manifest["StudyInstanceUID"].nunique()),
        "selected_diagnostic_cines": len(df_filtered),
        "max_clips_per_study": args.max_clips,
        "vasovision_filtering": not args.skip_vasovision,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disclaimer": "Research model only — not validated for clinical decision-making."
    }

    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Pipeline finished successfully in {elapsed_time}s.")
    logger.info(f"Results saved to: {output_dir}")
    print("\n" + BANNER)
    print(f"Summary of predictions saved to: {output_dir / 'summary.csv'}")
    print(f"Full raw outputs saved to:       {output_dir / 'predictions_raw.csv'}")
    print(f"Structured JSON saved to:        {output_dir / 'predictions.json'}")
    print(f"Metadata saved to:               {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
