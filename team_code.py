#!/usr/bin/env python
# team_code.py  –  PhysioNet / George B. Moody Challenge 2026
# PSG-based Cognitive Impairment Prediction
# Approach: Clinical feature engineering from CAISR annotations + demographics → XGBoost ensemble

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from helper_code import (
    find_patients,
    load_demographics,
    load_edf_to_nparrays,
    load_label,
    load_age,
    load_sex,
    load_bmi,
    get_standardized_race,
    get_standardized_ethnicity,
    update_demographics_table,
    HEADERS,
    DEMOGRAPHICS_FILE,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MODEL_FILENAME   = "xgb_moody2026.joblib"
IMPUTER_FILENAME = "imputer.joblib"
SCALER_FILENAME  = "scaler.joblib"

# CAISR sleep-stage encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
STAGE_N3   = 1
STAGE_N2   = 2
STAGE_N1   = 3
STAGE_REM  = 4
STAGE_WAKE = 5

# CAISR respiratory event encoding
RESP_OA = 1   # Obstructive Apnea
RESP_CA = 2   # Central Apnea
RESP_MA = 3   # Mixed Apnea
RESP_HY = 4   # Hypopnea
RESP_RERA = 5

# CAISR limb movement encoding
LIMB_ISOLATED = 1
LIMB_PERIODIC = 2

EPOCH_SEC = 30   # standard 30-second sleep staging epoch


# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def _safe_div(a, b, default=0.0):
    """Division that returns `default` when denominator is 0 or NaN."""
    try:
        if b == 0 or np.isnan(b):
            return default
        return a / b
    except Exception:
        return default


def extract_caisr_features(annot_edf_path: str) -> dict:
    """
    Extract sleep-architecture and clinical-event features from a CAISR
    algorithmic-annotation EDF file.

    Returns a flat dict of floats.  All values are NaN-safe.
    """
    feats = {}

    channel_dict, fs_dict = load_edf_to_nparrays(annot_edf_path)
    if not channel_dict:
        return feats   # empty → imputer will fill later

    # ── Helper: find a channel by partial name match ──────────────────────
    def _get(partial_name):
        for k in channel_dict:
            if partial_name.lower() in k.lower():
                return channel_dict[k], fs_dict.get(k, 1.0)
        return None, None

    # ── 1. Sleep Staging (30-second epochs) ──────────────────────────────
    stage_sig, stage_fs = _get('stage_caisr')
    if stage_sig is not None and len(stage_sig) > 0:
        stages = stage_sig.astype(int)
        valid  = stages[stages != 9]      # exclude Unavailable
        total_epochs = len(valid)
        total_sleep_min = _safe_div(
            np.sum(valid != STAGE_WAKE) * EPOCH_SEC, 60.0
        )
        tst_epochs = np.sum(valid != STAGE_WAKE)

        feats['sleep_efficiency']   = _safe_div(tst_epochs, total_epochs)
        feats['pct_n3']             = _safe_div(np.sum(valid == STAGE_N3),   tst_epochs)
        feats['pct_n2']             = _safe_div(np.sum(valid == STAGE_N2),   tst_epochs)
        feats['pct_n1']             = _safe_div(np.sum(valid == STAGE_N1),   tst_epochs)
        feats['pct_rem']            = _safe_div(np.sum(valid == STAGE_REM),  tst_epochs)
        feats['pct_wake']           = _safe_div(np.sum(valid == STAGE_WAKE), total_epochs)
        feats['total_sleep_min']    = total_sleep_min
        feats['total_epochs']       = float(total_epochs)

        # REM latency: epochs from sleep onset to first REM epoch
        sleep_onset_idx = next((i for i, s in enumerate(valid) if s != STAGE_WAKE), None)
        if sleep_onset_idx is not None:
            post_onset = valid[sleep_onset_idx:]
            rem_idx = next((i for i, s in enumerate(post_onset) if s == STAGE_REM), None)
            feats['rem_latency_epochs'] = float(rem_idx) if rem_idx is not None else float(total_epochs)
        else:
            feats['rem_latency_epochs'] = float(total_epochs)

        # WASO: Wake After Sleep Onset (epochs)
        if sleep_onset_idx is not None:
            feats['waso_epochs'] = float(
                np.sum(valid[sleep_onset_idx:] == STAGE_WAKE)
            )
        else:
            feats['waso_epochs'] = 0.0

        # Stage transition count (proxy for fragmentation)
        transitions = np.sum(np.diff(valid) != 0)
        feats['stage_transitions'] = float(transitions)
        feats['fragmentation_index'] = _safe_div(transitions, tst_epochs)

        # CAISR softmax probability features (mean uncertainty per stage)
        for prob_ch in ['caisr_prob_n3', 'caisr_prob_n2', 'caisr_prob_n1',
                        'caisr_prob_r', 'caisr_prob_w']:
            sig, _ = _get(prob_ch)
            if sig is not None and len(sig) > 0:
                feats[f'mean_{prob_ch}'] = float(np.nanmean(sig))
                feats[f'std_{prob_ch}']  = float(np.nanstd(sig))
            else:
                feats[f'mean_{prob_ch}'] = np.nan
                feats[f'std_{prob_ch}']  = np.nan

    # ── 2. Arousal Features ────────────────────────────────────────────────
    arousal_sig, arousal_fs = _get('arousal_caisr')
    total_sleep_min_local = feats.get('total_sleep_min', 0.0)

    if arousal_sig is not None and len(arousal_sig) > 0 and arousal_fs:
        # CAISR arousal at 0.5-second resolution
        resolution = 0.5
        arousal_events = 0
        in_event = False
        for val in arousal_sig:
            if val == 1 and not in_event:
                arousal_events += 1
                in_event = True
            elif val != 1:
                in_event = False

        feats['arousal_count']   = float(arousal_events)
        feats['arousal_index']   = _safe_div(arousal_events, total_sleep_min_local / 60.0)  # per hour
        feats['pct_arousal']     = _safe_div(np.sum(arousal_sig == 1), len(arousal_sig))

        # CAISR arousal probability stats
        prob_ar_sig, _ = _get('caisr_prob_arousal')
        if prob_ar_sig is not None:
            feats['mean_arousal_prob'] = float(np.nanmean(prob_ar_sig))
            feats['std_arousal_prob']  = float(np.nanstd(prob_ar_sig))

    # ── 3. Respiratory Event Features ─────────────────────────────────────
    resp_sig, resp_fs = _get('resp_caisr')
    if resp_sig is not None and len(resp_sig) > 0 and resp_fs:
        # 1-second resolution
        resp = resp_sig.astype(int)
        sleep_hr = _safe_div(total_sleep_min_local, 60.0, default=1.0)

        oa_count   = _count_events(resp, RESP_OA)
        ca_count   = _count_events(resp, RESP_CA)
        ma_count   = _count_events(resp, RESP_MA)
        hy_count   = _count_events(resp, RESP_HY)
        rera_count = _count_events(resp, RESP_RERA)
        total_resp = oa_count + ca_count + ma_count + hy_count + rera_count

        feats['ahi']              = _safe_div(oa_count + ca_count + ma_count + hy_count, sleep_hr)
        feats['oai']              = _safe_div(oa_count,   sleep_hr)  # Obstructive Apnea Index
        feats['cai']              = _safe_div(ca_count,   sleep_hr)  # Central Apnea Index
        feats['hi']               = _safe_div(hy_count,   sleep_hr)  # Hypopnea Index
        feats['rera_index']       = _safe_div(rera_count, sleep_hr)
        feats['rdi']              = _safe_div(total_resp, sleep_hr)  # Respiratory Disturbance Index
        feats['central_fraction'] = _safe_div(ca_count, total_resp)  # Central apnea fraction
        feats['total_resp_events'] = float(total_resp)

    # ── 4. Limb Movement Features ─────────────────────────────────────────
    limb_sig, limb_fs = _get('limb_caisr')
    if limb_sig is not None and len(limb_sig) > 0 and limb_fs:
        limb = limb_sig.astype(int)
        sleep_hr = _safe_div(total_sleep_min_local, 60.0, default=1.0)

        isolated_lm = _count_events(limb, LIMB_ISOLATED)
        periodic_lm = _count_events(limb, LIMB_PERIODIC)

        feats['plmi']  = _safe_div(periodic_lm,  sleep_hr)   # Periodic Limb Movement Index
        feats['lmi']   = _safe_div(isolated_lm,  sleep_hr)   # Limb Movement Index
        feats['total_plm'] = float(periodic_lm)

    return feats


def _count_events(signal: np.ndarray, event_code: int) -> int:
    """Count contiguous runs of `event_code` in a 1-second-resolution signal."""
    count = 0
    in_event = False
    for v in signal:
        if v == event_code and not in_event:
            count += 1
            in_event = True
        elif v != event_code:
            in_event = False
    return count


def extract_demographic_features(demo_data: dict) -> dict:
    """
    Convert raw demographics dict into numeric features.
    """
    feats = {}

    age = load_age(demo_data)
    feats['age'] = age if age > 0 else np.nan

    sex_str = load_sex(demo_data)
    feats['sex_female'] = 1.0 if sex_str == 'Female' else (0.0 if sex_str == 'Male' else np.nan)

    bmi = load_bmi(demo_data)
    feats['bmi'] = bmi if bmi > 0 else np.nan

    # BMI categories (clinical thresholds)
    if not np.isnan(feats.get('bmi', np.nan)):
        feats['obese']      = 1.0 if feats['bmi'] >= 30 else 0.0
        feats['overweight'] = 1.0 if 25 <= feats['bmi'] < 30 else 0.0
    else:
        feats['obese']      = np.nan
        feats['overweight'] = np.nan

    # Race one-hot
    race = get_standardized_race(demo_data)
    for r in ['White', 'Black', 'Asian', 'Others']:
        feats[f'race_{r.lower()}'] = 1.0 if race == r else 0.0

    # Ethnicity
    eth = get_standardized_ethnicity(demo_data)
    feats['hispanic'] = 1.0 if eth == 'Hispanic' else 0.0

    return feats


def build_feature_vector(demo_data: dict, annot_edf_path: str) -> dict:
    """
    Combine demographic and CAISR-annotation features into a single dict.
    Missing CAISR file → only demographic features (filled with NaN later).
    """
    feats = {}
    feats.update(extract_demographic_features(demo_data))

    if annot_edf_path and os.path.isfile(annot_edf_path):
        feats.update(extract_caisr_features(annot_edf_path))

    return feats


def dicts_to_matrix(feat_dicts: list, feature_names: list) -> np.ndarray:
    """Convert list of feature dicts → numpy matrix using a fixed column order."""
    rows = []
    for d in feat_dicts:
        rows.append([d.get(k, np.nan) for k in feature_names])
    return np.array(rows, dtype=np.float32)


def _find_annot_path(data_folder: str, site_id: str, bids_folder: str, session_id) -> str:
    """
    Construct the algorithmic annotation EDF path following the challenge
    directory structure:
      <data_folder>/algorithmic_annotations/<SiteID>/<BidsFolder>-ses<SessionID>.edf
    """
    annot_dir = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, str(site_id))
    fname = f"{bids_folder}-ses{session_id}.edf"
    candidate = os.path.join(annot_dir, fname)
    if os.path.isfile(candidate):
        return candidate

    # Fallback: search for partial match (in case of zero-padding differences)
    if os.path.isdir(annot_dir):
        for f in os.listdir(annot_dir):
            if bids_folder in f and f.endswith('.edf'):
                return os.path.join(annot_dir, f)

    return ""


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def train_model(data_folder: str, model_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"[train] Loading patient list from: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    if verbose:
        print(f"[train] Found {len(patients)} patient records.")

    feat_dicts = []
    labels     = []

    for i, patient in enumerate(patients):
        bids_folder = patient[HEADERS['bids_folder']]
        site_id     = patient[HEADERS['site_id']]
        session_id  = patient[HEADERS['session_id']]

        # Load demographics
        demo = load_demographics(demographics_file, bids_folder, session_id)
        label = load_label(demo)

        # Locate annotation EDF
        annot_path = _find_annot_path(data_folder, site_id, bids_folder, session_id)

        if verbose and (i % 50 == 0):
            annot_status = "found" if annot_path else "MISSING"
            print(f"  [{i+1}/{len(patients)}] {bids_folder} | label={label} | annot={annot_status}")

        fv = build_feature_vector(demo, annot_path)
        feat_dicts.append(fv)
        labels.append(label)

    if not feat_dicts:
        raise RuntimeError("No training data could be loaded.")

    # Determine stable feature column order
    all_keys = set()
    for d in feat_dicts:
        all_keys.update(d.keys())
    feature_names = sorted(all_keys)

    X = dicts_to_matrix(feat_dicts, feature_names)
    y = np.array(labels, dtype=np.int32)

    if verbose:
        pos = int(np.sum(y))
        print(f"[train] Features: {X.shape[1]} | Samples: {X.shape[0]} | Positives: {pos} ({100*pos/len(y):.1f}%)")

    # Pipeline: impute → scale → XGBoost
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    X_imp  = imputer.fit_transform(X)
    X_sc   = scaler.fit_transform(X_imp)

    # Class imbalance weight
    neg, pos = int(np.sum(y == 0)), int(np.sum(y == 1))
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators       = 500,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.7,
        min_child_weight   = 5,
        gamma              = 1.0,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos_weight,
        random_state       = 42,
        eval_metric        = 'auc',
        use_label_encoder  = False,
        n_jobs             = -1,
    )
    model.fit(X_sc, y)

    # Save artefacts
    joblib.dump(model,         os.path.join(model_folder, MODEL_FILENAME))
    joblib.dump(imputer,       os.path.join(model_folder, IMPUTER_FILENAME))
    joblib.dump(scaler,        os.path.join(model_folder, SCALER_FILENAME))
    joblib.dump(feature_names, os.path.join(model_folder, 'feature_names.joblib'))

    if verbose:
        print(f"[train] ✅ Model saved to: {model_folder}")


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def load_model(model_folder: str, verbose: int = 1):
    model         = joblib.load(os.path.join(model_folder, MODEL_FILENAME))
    imputer       = joblib.load(os.path.join(model_folder, IMPUTER_FILENAME))
    scaler        = joblib.load(os.path.join(model_folder, SCALER_FILENAME))
    feature_names = joblib.load(os.path.join(model_folder, 'feature_names.joblib'))

    if verbose:
        print(f"[load] ✅ Model loaded from: {model_folder}")

    return model, imputer, scaler, feature_names


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

def run_model(data_folder: str, model_folder: str, output_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"[run] Loading model …")

    model, imputer, scaler, feature_names = load_model(model_folder, verbose)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    if verbose:
        print(f"[run] Running inference on {len(patients)} records …")

    os.makedirs(output_folder, exist_ok=True)

    results_dict = {}   # {bids_folder: (binary_label, probability)}

    for i, patient in enumerate(patients):
        bids_folder = patient[HEADERS['bids_folder']]
        site_id     = patient[HEADERS['site_id']]
        session_id  = patient[HEADERS['session_id']]

        demo = load_demographics(demographics_file, bids_folder, session_id)
        annot_path = _find_annot_path(data_folder, site_id, bids_folder, session_id)

        if verbose and (i % 50 == 0):
            print(f"  [{i+1}/{len(patients)}] {bids_folder}")

        fv = build_feature_vector(demo, annot_path)
        x  = np.array([[fv.get(k, np.nan) for k in feature_names]], dtype=np.float32)

        x_imp = imputer.transform(x)
        x_sc  = scaler.transform(x_imp)

        prob   = float(model.predict_proba(x_sc)[0, 1])
        binary = int(prob >= 0.5)

        results_dict[bids_folder] = (binary, round(prob, 6))

    # Write updated demographics CSV with predictions
    output_file = update_demographics_table(demographics_file, output_folder, results_dict)

    if verbose:
        print(f"[run] ✅ Predictions saved to: {output_file}")
        print(f"[run] Total subjects processed: {len(results_dict)}")
