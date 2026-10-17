# Dataset

The assignment specifies the **Heart Disease UCI Dataset**. This project uses the
Cleveland subset provided by the UCI Machine Learning Repository.

The downloader in `src/data.py` retrieves:

`https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`

Run:

```bash
python -m src.data
```

The raw file is saved as `data/raw/heart_disease.csv`.

The original target contains values 0 to 4. The project converts it to a binary target:

- 0 -> no heart disease
- 1, 2, 3, 4 -> heart disease

Missing values represented by `?` are converted to missing values and handled inside the
scikit-learn preprocessing pipeline. This avoids data leakage because preprocessing is
fitted only on training data during model training.
