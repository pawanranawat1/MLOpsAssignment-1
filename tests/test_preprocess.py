import pandas as pd

from src.preprocess import build_preprocessor


def test_preprocessor_handles_missing_values():
    data = pd.DataFrame(
        [
            {
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
        ]
    )
    transformer = build_preprocessor()
    result = transformer.fit_transform(data)
    assert result.shape[0] == 1
    assert result.shape[1] > 0
