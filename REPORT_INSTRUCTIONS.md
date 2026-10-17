1. Create and activate the virtual environment.
2. Install `requirements.txt`.
3. Run `python -m src.data`.
4. Run `python -m src.eda`.
5. Run `$env:MLFLOW_ALLOW_FILE_STORE="true"`
6. Run `python -m src.train`.
7. Run `pytest -q` and `flake8 src tests --max-line-length=100`.
8. Build/run Docker.
9. Deploy to Kubernetes.
10. Capture the required screenshots into `screenshots/`.

