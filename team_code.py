import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from helper_code import (
    find_patients, load_demographics, load_age, load_sex, load_bmi,
    load_label, load_edf_to_nparrays, update_demographics_table,
    DEMOGRAPHICS_FILE, PHYSIOLOGICAL_DATA_SUBFOLDER
)

MODEL_FILENAME = "xgb_moody2026.model"
SCALER_FILENAME = "scaler.save"


def _get_edf_path(data_folder, bids_folder, session_id):
    """Correct EDF path based on Challenge directory structure."""
    site_id = bids_folder.split('-')[1][:5]  # e.g. 'S0001'
    filename = f"{bids_folder}-ses{session_id}.edf"
    return os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, filename)


def _get_annotation_path(data_folder, bids_folder, session_id):
    """CAISR algorithmic annotation EDF path."""
    site_id = bids_folder.split('-')[1][:5]
    filename = f"{bids_folder}-ses{session_id}.edf"
    return os.path.join(data_folder, 'algorithmic_annotations', site_id, filename)


def extract_features(data_folder, patient_info):
    """
    Demographic + PSG signal features extract করে।
    CAISR annotations থেকে AHI, sleep stage features বের করে।
    """
    bids_folder = patient_info.get('BidsFolder', '')
    session_id = str(patient_info.get('SessionID', ''))

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    demo = load_demographics(demographics_file, bids_folder, session_id)

    # --- Demographic features (3) ---
    age = load_age(demo)
    sex = 1.0 if load_sex(demo) == 'Female' else 0.0
    bmi = load_bmi(demo)
    features = [age, sex, bmi]

    # --- CAISR annotation features ---
    ann_path = _get_annotation_path(data_folder, bids_folder, session_id)
    if os.path.exists(ann_path):
        try:
            channel_dict, fs_dict = load_edf_to_nparrays(ann_path)

            # Sleep stages (encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake)
            stage_ch = next((v for k, v in channel_dict.items() if 'stage' in k.lower()), None)
            if stage_ch is not None and len(stage_ch) > 0:
                total = len(stage_ch)
                n3_pct  = np.sum(stage_ch == 1) / total
                n2_pct  = np.sum(stage_ch == 2) / total
                n1_pct  = np.sum(stage_ch == 3) / total
                rem_pct = np.sum(stage_ch == 4) / total
                wake_pct= np.sum(stage_ch == 5) / total
                features += [n3_pct, n2_pct, n1_pct, rem_pct, wake_pct]
            else:
                features += [0.0] * 5

            # Respiratory events → AHI proxy
            # (0=None,1=OA,2=CA,3=MA,4=HY,5=RERA)
            resp_ch = next((v for k, v in channel_dict.items() if 'resp' in k.lower() or 'apnea' in k.lower()), None)
            if resp_ch is not None and len(resp_ch) > 0:
                study_hours = len(resp_ch) / 3600.0
                ahi = np.sum(resp_ch > 0) / max(study_hours, 0.1)
                oa_count  = np.sum(resp_ch == 1) / max(study_hours, 0.1)
                ca_count  = np.sum(resp_ch == 2) / max(study_hours, 0.1)
                hy_count  = np.sum(resp_ch == 4) / max(study_hours, 0.1)
                features += [ahi, oa_count, ca_count, hy_count]
            else:
                features += [0.0] * 4

            # Arousal index
            arousal_ch = next((v for k, v in channel_dict.items() if 'arousal' in k.lower()), None)
            if arousal_ch is not None and len(arousal_ch) > 0:
                study_hours = len(arousal_ch) / 7200.0  # 0.5s resolution
                arousal_index = np.sum(arousal_ch == 1) / max(study_hours, 0.1)
                features += [arousal_index]
            else:
                features += [0.0]

        except Exception as e:
            print(f"Annotation load error for {bids_folder}: {e}")
            features += [0.0] * 10
    else:
        features += [0.0] * 10

    # --- Raw EDF signal features (basic stats) ---
    edf_path = _get_edf_path(data_folder, bids_folder, session_id)
    if os.path.exists(edf_path):
        try:
            channel_dict, _ = load_edf_to_nparrays(edf_path)
            # EEG channel থেকে variance (brain activity proxy)
            eeg_ch = next((v for k, v in channel_dict.items()
                           if any(x in k.lower() for x in ['eeg', 'c3', 'c4', 'f3', 'f4', 'o1', 'o2'])), None)
            if eeg_ch is not None and len(eeg_ch) > 0:
                features += [float(np.var(eeg_ch)), float(np.percentile(eeg_ch, 95))]
            else:
                features += [0.0, 0.0]

            # ECG/heart rate channel
            ecg_ch = next((v for k, v in channel_dict.items()
                           if any(x in k.lower() for x in ['ecg', 'ekg', 'hr', 'heart'])), None)
            if ecg_ch is not None and len(ecg_ch) > 0:
                features += [float(np.mean(np.abs(ecg_ch))), float(np.std(ecg_ch))]
            else:
                features += [0.0, 0.0]
        except Exception as e:
            print(f"EDF load error for {bids_folder}: {e}")
            features += [0.0] * 4
    else:
        features += [0.0] * 4

    # Total: 3 + 10 + 4 = 17 features
    return np.array(features[:17], dtype=np.float32)


# ====================== TRAIN ======================

def train_model(data_folder, model_folder, verbose=1):
    if verbose:
        print(f"Training started: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    X_train, y_train = [], []
    for patient_info in patients:
        bids_folder = patient_info.get('BidsFolder', '')
        session_id = str(patient_info.get('SessionID', ''))
        demo = load_demographics(demographics_file, bids_folder, session_id)
        label = load_label(demo)
        features = extract_features(data_folder, patient_info)
        X_train.append(features)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    if verbose:
        print(f"Patients: {len(y_train)}, Positive: {sum(y_train)}, Negative: {len(y_train)-sum(y_train)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=max(1, (len(y_train)-sum(y_train)) / max(sum(y_train), 1))
    )
    model.fit(X_scaled, y_train)

    joblib.dump(model, os.path.join(model_folder, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(model_folder, SCALER_FILENAME))

    if verbose:
        print(f"Model saved: {model_folder}")


# ====================== LOAD ======================

def load_model(model_folder, verbose=1):
    model  = joblib.load(os.path.join(model_folder, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(model_folder, SCALER_FILENAME))
    if verbose:
        print(f"Model loaded from {model_folder}")
    return model, scaler


# ====================== RUN ======================

def run_model(data_folder, model_folder, output_folder, verbose=1):
    if verbose:
        print(f"Running inference: {data_folder}")

    model, scaler = load_model(model_folder, verbose)
    os.makedirs(output_folder, exist_ok=True)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    results_dict = {}
    for patient_info in patients:
        bids_folder = patient_info.get('BidsFolder', '')
        features = extract_features(data_folder, patient_info)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prob  = float(model.predict_proba(features_scaled)[0, 1])
        label = int(prob >= 0.5)
        results_dict[bids_folder] = (label, round(prob, 6))

    output_file = update_demographics_table(demographics_file, output_folder, results_dict)

    if verbose:
        print(f"Predictions saved: {output_file}")
        print(f"Total subjects: {len(results_dict)}")
