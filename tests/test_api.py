from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
