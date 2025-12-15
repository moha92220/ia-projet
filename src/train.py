import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE

TARGET = "MedHouseVal"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    raw_path = os.path.join(DATA_RAW_DIR, "california_housing_raw.csv")
    df = pd.read_csv(raw_path)

    # --- Basic cleaning (exemple)
    df = df.drop_duplicates()
    df = df.dropna()

    # Sauvegarde processed
    processed_path = os.path.join(DATA_PROCESSED_DIR, "california_housing_processed.csv")
    df.to_csv(processed_path, index=False)

    # --- Features / cible
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_features = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    best_name = None
    best_pipeline = None
    best_rmse = float("inf")

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        cur_rmse = rmse(y_test, preds)
        cur_r2 = float(r2_score(y_test, preds))

        results.append({"model": name, "rmse": cur_rmse, "r2": cur_r2})
        print(f"{name}: RMSE={cur_rmse:.4f} | R2={cur_r2:.4f}")

        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            best_name = name
            best_pipeline = pipeline

    # Save best model pipeline
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "best_rmse": best_rmse,
            "all_results": results
        }, f, indent=2)

    print(f"\n[OK] Best model: {best_name} (RMSE={best_rmse:.4f})")
    print(f"[OK] Saved -> {model_path}")
    print(f"[OK] Metrics -> {metrics_path}")
    print(f"[OK] Processed -> {processed_path}")

if __name__ == "__main__":
    main()
