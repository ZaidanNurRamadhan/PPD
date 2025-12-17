import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Paths
DATA_PATH = Path(__file__).resolve().parent / "cardio_train.csv"
MODEL_PATH = Path(__file__).resolve().parent / "ml" / "best_xgb_pipeline.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "ml" / "model_metadata.json"

# Feature schema (selaras dengan model_utils.py)
INPUT_FIELDS = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

NUMERIC_FEATURES = [
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "bmi",
    "map",
]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    height_m = df["height"] / 100
    df["bmi"] = (df["weight"] / (height_m**2)).round(2)
    df["map"] = ((2 * df["ap_lo"] + df["ap_hi"]) / 3).round(2)
    return df


def _prepare_training_frame(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = raw.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Konversi umur (hari) ke tahun
    if "age_years" not in df.columns and "age" in df.columns:
        df["age_years"] = (df["age"] / 365).astype(int)

    df = _compute_derived(df)

    # basic data cleaning (outlier/bounds)
    bounds = (
        (df["ap_hi"] > 50)
        & (df["ap_hi"] < 300)
        & (df["ap_lo"] > 30)
        & (df["ap_lo"] < 200)
        & (df["height"] > 120)
        & (df["height"] < 220)
        & (df["weight"] > 20)
        & (df["weight"] < 200)
        & (df["bmi"] > 10)
        & (df["bmi"] < 60)
    )
    df = df[bounds].reset_index(drop=True)

    features = df[INPUT_FIELDS + ["bmi", "map"]]
    target = df["cardio"]
    return features, target


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def train_model() -> Tuple[Pipeline, dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")
    X, y = _prepare_training_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = _build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_prob)), 4),
        "samples": len(df),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(classification_report(y_val, y_pred))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Metrics: {metrics}")
    return model, metrics


if __name__ == "__main__":
    train_model()
