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


def extract_features(data_folder: str, patient_info: dict) -> np.ndarray:
    """
    Patient এর demographic + basic EDF signal features extract করে।
    """
    bids_folder = patient_info.get('BidsFolder', '')
    session_id = patient_info.get('SessionID', '')

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    demo = load_demographics(demographics_file, bids_folder, session_id)

    age = load_age(demo)
    sex = 1.0 if load_sex(demo) == 'Female' else 0.0
    bmi = load_bmi(demo)

    # Demographic features
    features = [age, sex, bmi]

    # EDF signal থেকে basic stats বের করা
    edf_path = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
        bids_folder, session_id, f"{bids_folder}_{session_id}_PSG.edf"
    )

    if os.path.exists(edf_path):
        channel_dict, fs_dict = load_edf_to_nparrays(edf_path)
        for ch_name, signal in list(channel_dict.items())[:5]:  # প্রথম ৫টি channel
            if len(signal) > 0:
                features += [
                    float(np.mean(signal)),
                    float(np.std(signal)),
                    float(np.median(signal)),
                ]
            else:
                features += [0.0, 0.0, 0.0]
    else:
        features += [0.0] * 15  # EDF না পেলে zeros

    # সব features কে 18-length array এ প্যাড করা
    features = np.array(features[:18], dtype=np.float32)
    if len(features) < 18:
        features = np.pad(features, (0, 18 - len(features)))

    return features


# ====================== TRAIN ======================

def train_model(data_folder: str, model_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"Training started: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    X_train, y_train = [], []

    for patient_info in patients:
        bids_folder = patient_info.get('BidsFolder', '')
        session_id = patient_info.get('SessionID', '')
        demo = load_demographics(demographics_file, bids_folder, session_id)
        label = load_label(demo)

        features = extract_features(data_folder, patient_info)
        X_train.append(features)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    if verbose:
        print(f"Total patients: {len(y_train)}, Positive: {sum(y_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=max(1, (len(y_train) - sum(y_train)) / max(sum(y_train), 1))
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, os.path.join(model_folder, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(model_folder, SCALER_FILENAME))

    if verbose:
        print(f"Model saved in: {model_folder}")


# ====================== LOAD ======================

def load_model(model_folder: str, verbose: int = 1):
    model = joblib.load(os.path.join(model_folder, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(model_folder, SCALER_FILENAME))
    if verbose:
        print(f"Model loaded from {model_folder}")
    return model, scaler


# ====================== RUN ======================

def run_model(data_folder: str, model_folder: str, output_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"Running inference on: {data_folder}")

    model, scaler = load_model(model_folder, verbose)
    os.makedirs(output_folder, exist_ok=True)

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patients = find_patients(demographics_file)

    results_dict = {}

    for patient_info in patients:
        bids_folder = patient_info.get('BidsFolder', '')
        features = extract_features(data_folder, patient_info)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prob = float(model.predict_proba(features_scaled)[0, 1])
        label = int(prob >= 0.5)
        results_dict[bids_folder] = (label, round(prob, 6))

    # Official output format: demographics.csv তে prediction যোগ করা
    output_file = update_demographics_table(demographics_file, output_folder, results_dict)

    if verbose:
        print(f"Predictions saved to: {output_file}")
        print(f"Total subjects: {len(results_dict)}")
