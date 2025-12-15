import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from config import DATA_RAW_DIR

def main():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # contient features + cible (MedHouseVal)

    raw_path = os.path.join(DATA_RAW_DIR, "california_housing_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"[OK] Saved raw data -> {raw_path} | shape={df.shape}")

if __name__ == "__main__":
    main()
