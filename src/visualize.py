import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE

TARGET = "MedHouseVal"

def main():
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "california_housing_processed.csv"))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    y_pred = model.predict(X_test)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("True vs Predicted (Best Model)")
    fig_path = os.path.join(REPORTS_DIR, "figures", "true_vs_pred.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved figure -> {fig_path}")

if __name__ == "__main__":
    main()
