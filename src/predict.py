import os
import joblib
import pandas as pd
from config import MODELS_DIR

def main():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    model = joblib.load(model_path)

    # Exemple : une “nouvelle” ligne (features)
    sample = pd.DataFrame([{
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }])

    pred = model.predict(sample)[0]
    print(f"Prediction (MedHouseVal) = {pred:.4f}")

if __name__ == "__main__":
    main()
