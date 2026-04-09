#!/usr/bin/env python
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from helper_code import *

# Get script directory for robust path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

################################################################################
# REQUIRED FUNCTIONS - DO NOT CHANGE FUNCTION SIGNATURES
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Finding the Challenge data and extracting features...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    features = []
    labels = []

    pbar = tqdm(range(num_records), desc="Processing records", disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # Demographics features
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            demo_features = extract_demographic_features(patient_data)

            # Physiological (PSG) signals
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
            if os.path.exists(phys_file):
                phys_data, phys_fs = load_signal_data(phys_file)
                phys_features = extract_physiological_features(phys_data, phys_fs, csv_path)
            else:
                phys_features = np.zeros(49)  # safe fallback

            # Algorithmic annotations
            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
                algo_features = extract_algorithmic_annotations_features(algo_data)
            else:
                algo_features = np.zeros(12)

            combined_features = np.hstack([demo_features, phys_features, algo_features])

            label = load_diagnoses(patient_data_file, patient_id)

            if label in [0, 1]:
                features.append(combined_features)
                labels.append(label)

        except Exception as e:
            if verbose:
                print(f"Skipping record {i} due to error: {e}")
            continue

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=int)

    if len(features) == 0:
        raise ValueError("No valid training samples found!")

    if verbose:
        print(f'Training RandomForest on {len(features)} samples...')

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(features, labels)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print('Training completed successfully.\n')


def load_model(model_folder, verbose):
    model_path = os.path.join(model_folder, 'model.sav')
    saved_dict = joblib.load(model_path)
    return saved_dict['model']


def run_model(model, record, data_folder, verbose):
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demo_features = extract_demographic_features(patient_data)

    # Physiological signals
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        phys_features = extract_physiological_features(phys_data, phys_fs)
    else:
        phys_features = np.zeros(49)

    # Algorithmic annotations
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algo_features = extract_algorithmic_annotations_features(algo_data)
    else:
        algo_features = np.zeros(12)

    features = np.hstack([demo_features, phys_features, algo_features]).reshape(1, -1)

    binary_output = int(model.predict(features)[0])
    probability_output = float(model.predict_proba(features)[0][1])

    return binary_output, probability_output


def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


################################################################################
# OPTIONAL: Paste your custom feature functions here if you want to improve
# extract_demographic_features, extract_physiological_features, extract_algorithmic_annotations_features
# For now we will use the ones from helper_code or simple fallbacks.
################################################################################

