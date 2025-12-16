from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="ML Housing Price API")

model = joblib.load("models/best_model.joblib")

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {"status": "ok", "service": "ml-housing-api"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HousingInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
