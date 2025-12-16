from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="ML Housing Price API")

model = None

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "models/best_model.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully")
    else:
        print("Model not found, running without model (CI mode)")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HousingInput):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}
