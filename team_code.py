import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

from helper_code import (
    DEMOGRAPHICS_FILE,
    HEADERS,
    load_demographics,
    get_standardized_race,
    get_standardized_ethnicity,
)

MODEL_FILENAME = "xgb_moody2026.model"
SCALER_FILENAME = "scaler.save"
IMPUTER_FILENAME = "imputer.save"
FEATURES_FILENAME = "feature_columns.save"


def _build_features(df: pd.DataFrame, feature_cols=None):
    d = df.copy()

    # Numeric
    d["Age"] = pd.to_numeric(d.get("Age", np.nan), errors="coerce")
    d["BMI"] = pd.to_numeric(d.get("BMI", np.nan), errors="coerce")

    # Sex
    sex = d.get("Sex", pd.Series([""] * len(d))).astype(str).str.upper()
    d["Sex_M"] = (sex == "M").astype(float)
    d["Sex_F"] = (sex == "F").astype(float)

    # Standardized race
    race_std = d.apply(lambda row: get_standardized_race(row.to_dict()), axis=1)
    for race in ["White", "Black", "Asian", "Others", "Unavailable"]:
        d[f"Race_{race}"] = (race_std == race).astype(float)

    # Standardized ethnicity
    eth_std = d.apply(lambda row: get_standardized_ethnicity(row.to_dict()), axis=1)
    for eth in ["Hispanic", "Not Hispanic", "Unavailable"]:
        d[f"Ethnicity_{eth}"] = (eth_std == eth).astype(float)

    # Site: fixed from training data to keep columns stable
    known_sites = ["S0001", "I0002", "I0004", "I0006", "I0007"]
    site_series = d.get("SiteID", pd.Series([""] * len(d))).astype(str)
    for site in known_sites:
        d[f"Site_{site}"] = (site_series == site).astype(float)

    default_feature_cols = [
        "Age", "BMI",
        "Sex_M", "Sex_F",
        "Race_White", "Race_Black", "Race_Asian", "Race_Others", "Race_Unavailable",
        "Ethnicity_Hispanic", "Ethnicity_Not Hispanic", "Ethnicity_Unavailable",
        "Site_S0001", "Site_I0002", "Site_I0004", "Site_I0006", "Site_I0007",
    ]

    if feature_cols is None:
        feature_cols = default_feature_cols

    for col in feature_cols:
        if col not in d.columns:
            d[col] = 0.0

    X = d[feature_cols].astype(np.float32).values
    return X, feature_cols


def train_model(data_folder: str, model_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"Training started: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    demo_path = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"{DEMOGRAPHICS_FILE} not found at {demo_path}")

    df = pd.read_csv(demo_path)

    if verbose:
        print(f"Loaded {len(df)} records from demographics.csv")
        print(f"Columns: {list(df.columns)}")

    if "Cognitive_Impairment" not in df.columns:
        raise ValueError("Cognitive_Impairment column not found!")

    df["label"] = df["Cognitive_Impairment"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
    )

    df_labeled = df[df["label"].notna()].copy()

    if verbose:
        print(f"Labeled samples: {len(df_labeled)} | Positive: {int(df_labeled['label'].sum())}")

    X, feature_cols = _build_features(df_labeled)
    y = df_labeled["label"].values.astype(np.int32)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
    )
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_folder, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(model_folder, SCALER_FILENAME))
    joblib.dump(imputer, os.path.join(model_folder, IMPUTER_FILENAME))
    joblib.dump(feature_cols, os.path.join(model_folder, FEATURES_FILENAME))

    if verbose:
        print(f"Top features trained. Model saved to {model_folder}")


def load_model(model_folder: str, verbose: int = 1):
    model = joblib.load(os.path.join(model_folder, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(model_folder, SCALER_FILENAME))
    imputer = joblib.load(os.path.join(model_folder, IMPUTER_FILENAME))
    feature_cols = joblib.load(os.path.join(model_folder, FEATURES_FILENAME))

    if verbose:
        print(f"Model loaded from {model_folder}")

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_cols": feature_cols,
    }


def run_model(model, record, data_folder: str, verbose: int = 1):
    if verbose:
        print(f"Running inference for record: {record}")

    metadata_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_id = record[HEADERS["bids_folder"]]
    session_id = record[HEADERS["session_id"]]

    data = load_demographics(metadata_file, patient_id, session_id)
    if not data:
        raise ValueError(f"No demographics found for patient={patient_id}, session={session_id}")

    df = pd.DataFrame([data])

    X, _ = _build_features(df, model["feature_cols"])
    X = model["imputer"].transform(X)
    X = model["scaler"].transform(X)

    prob = float(model["model"].predict_proba(X)[0, 1])
    binary = int(prob >= 0.5)

    return binary, prob