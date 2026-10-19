"""Run one prediction from a JSON file."""

import argparse
import json
import pandas as pd
import joblib

from src.config import MODEL_PATH


def predict_from_json(input_path: str):
    model = joblib.load(MODEL_PATH)
    data = json.loads(open(input_path, encoding="utf-8").read())
    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": "heart disease risk" if prediction == 1 else "no heart disease risk",
        "disease_probability": round(probability, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    parser.add_argument("input_json", help="Path to patient JSON file")
    args = parser.parse_args()
    print(json.dumps(predict_from_json(args.input_json), indent=2))
