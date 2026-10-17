from fastapi.testclient import TestClient

from src import api


class DummyModel:
    def predict(self, rows):
        return [1]

    def predict_proba(self, rows):
        return [[0.15, 0.85]]


def test_health_endpoint_without_model():
    original = api.model
    api.model = None
    client = TestClient(api.app)
    response = client.get("/health")
    api.model = original
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_endpoint():
    original = api.model
    api.model = DummyModel()
    client = TestClient(api.app)
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": None,
        "thal": None,
    }
    response = client.post("/predict", json=payload)
    api.model = original
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
