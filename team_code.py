import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

MODEL_FILENAME = "xgb_moody2026.model"
SCALER_FILENAME = "scaler.save"
IMPUTER_FILENAME = "imputer.save"

def extract_features(demographics_df):
    """Demographics থেকে features বের করো"""
    df = demographics_df.copy()
    
    # Numeric features
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    
    # Sex encoding
    df['Sex_M'] = (df['Sex'].str.upper() == 'M').astype(float)
    df['Sex_F'] = (df['Sex'].str.upper() == 'F').astype(float)
    
    # Race encoding
    for race in ['White', 'Black', 'Asian', 'Others']:
        df[f'Race_{race}'] = (df['Race'] == race).astype(float)
    
    # Ethnicity encoding
    df['Hispanic'] = (df['Ethnicity'] == 'Hispanic').astype(float)
    
    # Site encoding
    for site in df['SiteID'].unique():
        df[f'Site_{site}'] = (df['SiteID'] == site).astype(float)
    
    feature_cols = ['Age', 'BMI', 'Sex_M', 'Sex_F',
                    'Race_White', 'Race_Black', 'Race_Asian', 'Race_Others',
                    'Hispanic'] + [c for c in df.columns if c.startswith('Site_')]
    
    # শুধু যে columns আছে সেগুলো নাও
    available = [c for c in feature_cols if c in df.columns]
    return df[available].values.astype(np.float32), df.index.tolist()

def train_model(data_folder: str, model_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"Training started: {data_folder}")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # demographics.csv লোড করো
    demo_path = os.path.join(data_folder, 'demographics.csv')
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"demographics.csv not found at {demo_path}")
    
    df = pd.read_csv(demo_path)
    if verbose:
        print(f"Loaded {len(df)} records from demographics.csv")
        print(f"Columns: {list(df.columns)}")
    
    # Label column
    if 'Cognitive_Impairment' not in df.columns:
        raise ValueError("Cognitive_Impairment column not found!")
    
    # TRUE/FALSE → 1/0
    df['label'] = df['Cognitive_Impairment'].map({True: 1, False: 0, 
                                                    'TRUE': 1, 'FALSE': 0,
                                                    'True': 1, 'False': 0})
    
    # লেবেল আছে এমন rows রাখো
    df_labeled = df[df['label'].notna()].copy()
    if verbose:
        print(f"Labeled samples: {len(df_labeled)} | Positive: {df_labeled['label'].sum():.0f}")
    
    X, _ = extract_features(df_labeled)
    y = df_labeled['label'].values.astype(np.int32)
    
    # Impute + Scale
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Class imbalance handle
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
        eval_metric='auc',
        use_label_encoder=False
    )
    model.fit(X_scaled, y)
    
    joblib.dump(model, os.path.join(model_folder, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(model_folder, SCALER_FILENAME))
    joblib.dump(imputer, os.path.join(model_folder, IMPUTER_FILENAME))
    
    # Feature importance
    if verbose:
        fi = model.feature_importances_
        print(f"Top features trained. Model saved to {model_folder}")

def load_model(model_folder: str, verbose: int = 1):
    model = joblib.load(os.path.join(model_folder, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(model_folder, SCALER_FILENAME))
    imputer = joblib.load(os.path.join(model_folder, IMPUTER_FILENAME))
    if verbose:
        print(f"Model loaded from {model_folder}")
    return model, scaler, imputer

def run_model(data_folder: str, model_folder: str, output_folder: str, verbose: int = 1) -> None:
    if verbose:
        print(f"Running inference on: {data_folder}")
    
    model, scaler, imputer = load_model(model_folder, verbose)
    os.makedirs(output_folder, exist_ok=True)
    
    demo_path = os.path.join(data_folder, 'demographics.csv')
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"demographics.csv not found at {demo_path}")
    
    df = pd.read_csv(demo_path)
    if verbose:
        print(f"Loaded {len(df)} test records")
    
    X, indices = extract_features(df)
    X = imputer.transform(X)
    X_scaled = scaler.transform(X)
    
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Output format: Challenge এর evaluate_model.py যা expect করে
    results = pd.DataFrame({
        'id': df['BDSPPatientID'] if 'BDSPPatientID' in df.columns else df.index,
        'Cognitive_Impairment': probabilities.round(6)
    })
    
    output_file = os.path.join(output_folder, 'predictions.csv')
    results.to_csv(output_file, index=False)
    
    if verbose:
        print(f"Predictions saved: {output_file}")
        print(f"Total: {len(results)} | Mean prob: {probabilities.mean():.4f}")
