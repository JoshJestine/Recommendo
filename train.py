"""
Train an XGBoost regressor to predict Monthly_Premium_USD from driver +
vehicle features.

Usage:
    python train.py

Outputs:
    artifacts/model.json           — trained XGBoost model
    artifacts/feature_schema.json  — column order, dtypes, and category levels
    artifacts/vehicle_catalog.csv  — unique (Make, Model) pairs from training data
    artifacts/state_list.json      — list of states
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "auto_insurance_premiums.csv"
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Feature configuration
# -----------------------------------------------------------------------------
TARGET = "Monthly_Premium_USD"
DROP_COLS = ["Driver_ID", "State_Abbr"]  # IDs / redundant

CATEGORICAL_COLS = [
    "Gender",
    "State",
    "Location",
    "Make",
    "Model",
    "Ownership",
]
NUMERIC_COLS = [
    "Age",
    "Credit_Score",
    "Model_Year",
    "Previous_Claims",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Place auto_insurance_premiums.csv in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas Categorical dtype so XGBoost can
    consume them natively (enable_categorical=True)."""
    df = df.copy()
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c])
    return df


def main():
    print(f"Loading {DATA_PATH} ...")
    df = load_data()
    print(f"Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    df = prepare_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost ...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print("\n=== Test set performance ===")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAE   : ${mae:.2f}")
    print(f"  MAPE  : {mape:.2f}%")

    # ---- Save artifacts ----
    model.save_model(ART_DIR / "model.json")
    print(f"\nSaved model       → {ART_DIR/'model.json'}")

    # Feature schema (column order + category levels so inference matches training)
    schema = {
        "feature_order": list(X.columns),
        "numeric": NUMERIC_COLS,
        "categorical": {
            c: list(map(str, df[c].cat.categories)) for c in CATEGORICAL_COLS
        },
    }
    with open(ART_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema      → {ART_DIR/'feature_schema.json'}")

    # Unique (Make, Model) catalog for inference
    catalog = (
        df[["Make", "Model"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["Make", "Model"])
        .reset_index(drop=True)
    )
    catalog.to_csv(ART_DIR / "vehicle_catalog.csv", index=False)
    print(f"Saved catalog     → {ART_DIR/'vehicle_catalog.csv'}  ({len(catalog)} vehicles)")

    # State list for the dropdown
    states = sorted(df["State"].astype(str).unique().tolist())
    with open(ART_DIR / "state_list.json", "w") as f:
        json.dump(states, f, indent=2)
    print(f"Saved state list  → {ART_DIR/'state_list.json'}  ({len(states)} states)")

    # Feature importance (just print top 10)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("\nTop 10 feature importances:")
    for name, val in importances.head(10).items():
        print(f"  {name:18s}  {val:.4f}")


if __name__ == "__main__":
    main()
