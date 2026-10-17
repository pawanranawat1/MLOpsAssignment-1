# MLOps Assignment 01 - Heart Disease Prediction

This project implements the assignment as an end-to-end MLOps solution:

**UCI dataset -> EDA -> preprocessing -> two models -> tuning -> MLflow -> tests -> CI/CD -> Docker -> Kubernetes -> API monitoring**

The implementation is intentionally written in small Python files with comments so that a beginner can follow each stage.

> Important: the assignment's academic-integrity section asks students to understand and personalize their work. Treat this repository as a learning/reference implementation, run every step yourself, inspect the outputs, and personalize the final report/screenshots/video.

---

## 1. What the assignment asks for

The supplied assignment asks for data acquisition and EDA, feature engineering and at least two classifiers, MLflow experiment tracking, model packaging, unit tests and CI/CD, Docker API serving, Kubernetes/cloud deployment, monitoring/logging, and a 10-page report. It also asks for a reproducible clean setup and Docker proof. See the supplied assignment for the official wording. 

---

## 2. Project structure

```text
heart_disease_mlops/
│
├── data/
│   ├── raw/                 # downloaded UCI data goes here
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py            # paths and feature names
│   ├── data.py              # download + clean dataset
│   ├── preprocess.py        # imputation, scaling, one-hot encoding
│   ├── eda.py               # EDA charts
│   ├── train.py             # training, tuning, evaluation, MLflow
│   ├── predict.py           # command-line prediction
│   └── api.py               # FastAPI service
│
├── tests/
│   ├── test_preprocess.py
│   └── test_api.py
│
├── notebooks/
│   └── 01_eda_and_training.ipynb
│
├── models/                  # trained model is created here
├── reports/figures/         # EDA + model plots are created here
├── mlruns/                  # local MLflow tracking data is created here
├── k8s/deployment.yaml      # Kubernetes Deployment + LoadBalancer Service
├── .github/workflows/ci.yml # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── sample_request.json
├── REPORT.docx
├── REPORT.pdf
└── this README
```

---

# 3. Prerequisites

Recommended:

- Windows 10/11, Linux, or macOS
- Python 3.11
- Git
- Docker Desktop
- Optional: Kubernetes enabled in Docker Desktop OR Minikube
- Optional: VS Code / PyCharm / Jupyter
- GitHub account for CI/CD

Check installations:

```powershell
python --version
git --version
docker --version
kubectl version --client
```

Python 3.11 is used in the Docker image and GitHub Actions workflow to keep the environment predictable.

---

# 4. Stage 1 - Create the project environment

Open PowerShell in the project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install all dependencies:

```powershell
pip install -r requirements.txt
```

Confirm:

```powershell
python -c "import pandas, sklearn, mlflow, fastapi; print('All main packages are installed')"
```

---

# 5. Stage 2 - Download the UCI dataset

Run:

```powershell
python -m src.data
```

You should see the number of rows and a preview of the data.

The script:

1. Downloads the Cleveland UCI data.
2. Adds readable column names.
3. Converts `?` to missing values.
4. Converts columns to numeric values.
5. Converts the original target 0-4 into a binary target.

If your network blocks the UCI URL, download `processed.cleveland.data` manually from the UCI Machine Learning Repository and save it as:

```text
data/raw/heart_disease.csv
```

Then run:

```powershell
python -c "from src.data import load_dataset; print(load_dataset().head())"
```

---

# 6. Stage 3 - Exploratory Data Analysis

Run:

```powershell
python -m src.eda
```

The following files will be created:

```text
reports/figures/class_balance.png
reports/figures/numeric_histograms.png
reports/figures/correlation_heatmap.png
reports/figures/age_vs_thalach.png
```

What to explain in your report:

- **Class balance:** shows how many patients are in each target class.
- **Histograms:** show distributions of age, blood pressure, cholesterol, maximum heart rate, and oldpeak.
- **Correlation heatmap:** helps identify relationships between numerical variables.
- **Age vs maximum heart rate:** provides a simple visual comparison of two clinically relevant numerical features.

Do not write exact numerical conclusions until you have run the script and inspected the actual plots.

---

# 7. Stage 4 - Preprocessing and feature engineering

The project uses a single scikit-learn `ColumnTransformer` so training and inference use exactly the same preprocessing.

### Numeric features

- age
- trestbps
- chol
- thalach
- oldpeak

Processing:

1. Missing values -> median imputation.
2. Values -> standard scaling.

### Categorical/discrete features

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

Processing:

1. Missing values -> most-frequent imputation.
2. Values -> one-hot encoding.

This is packaged together with the model in one Pipeline, which is important for reproducibility and prevents accidentally applying different transformations at serving time.

---

# 8. Stage 5 - Train two classification models

Run:

```powershell
python -m src.train
```

The training script:

1. Loads the cleaned dataset.
2. Splits it into 80% training and 20% test data using stratification.
3. Builds a Logistic Regression pipeline.
4. Builds a Random Forest pipeline.
5. Evaluates both models using accuracy, precision, recall, and ROC-AUC.
6. Tunes Random Forest with 5-fold Stratified Cross-Validation.
7. Selects the tuned Random Forest model.
8. Fits the selected pipeline on all available data.
9. Saves the model to `models/heart_disease_model.joblib`.
10. Saves metadata to `models/model_metadata.json`.
11. Saves comparison and evaluation plots.
12. Logs experiments to local MLflow.

### Quick training without MLflow

For a fast code check:

```powershell
python -m src.train --skip-mlflow
```

For the actual assignment run, use the normal command so the experiment is tracked.

---

# 9. Stage 6 - MLflow experiment tracking

After training, start the local MLflow UI:

```powershell
python -m mlflow ui --backend-store-uri .\mlruns --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

You should see the experiment:

```text
heart-disease-classification
```

Expected runs:

- Logistic Regression
- Random Forest
- Random Forest Tuned

Tracked information includes model name, split settings, metrics, tuned parameters, ROC plots, and confusion matrices.

Take a screenshot of the MLflow experiment page for the report.

---

# 10. Stage 7 - Unit tests

Run:

```powershell
pytest -q
```

The tests check:

- The preprocessing pipeline can handle missing values.
- The API health endpoint works.
- The prediction endpoint accepts a valid patient payload and returns a prediction.

Run linting:

```powershell
flake8 src tests --max-line-length=100
```

Both commands should pass before pushing to GitHub.

---

# 11. Stage 8 - Command-line inference

After training, run:

```powershell
python -m src.predict sample_request.json
```

The output will contain:

```json
{
  "prediction": 0,
  "prediction_label": "...",
  "disease_probability": 0.1234
}
```

The exact number is intentionally generated by your trained model, so do not hard-code it in your report.

---

# 12. Stage 9 - Run the FastAPI locally

Start the API:

```powershell
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Check health:

```text
http://127.0.0.1:8000/health
```

Check Prometheus metrics:

```text
http://127.0.0.1:8000/metrics
```

### Test `/predict`

Use the Swagger UI and send the contents of `sample_request.json`.

Or use PowerShell:

```powershell
$body = Get-Content .\sample_request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

The response contains prediction, prediction label, confidence, and disease probability.

> This is an educational ML classifier, not a medical diagnostic system. A model prediction must not be treated as medical advice.

---

# 13. Stage 10 - Docker containerization

First make sure the trained model exists:

```powershell
Test-Path .\models\heart_disease_model.joblib
```

It should return `True`.

Build:

```powershell
docker build -t heart-disease-api:latest .
```

Run:

```powershell
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Test prediction using `sample_request.json` as shown in the API section.

### Docker Compose alternative

```powershell
docker compose up --build
```

Stop it with:

```powershell
docker compose down
```

Take screenshots of:

1. Successful Docker build.
2. Running container.
3. Swagger `/predict` response.

---

# 14. Stage 11 - GitHub Actions CI/CD

Create a GitHub repository and push this project.

```powershell
git init
git add .
git commit -m "Initial MLOps assignment implementation"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

The workflow in `.github/workflows/ci.yml` runs:

1. Checkout.
2. Python setup.
3. Dependency installation.
4. Flake8 linting.
5. Dataset download.
6. Pytest.
7. Model training.
8. Docker image build.
9. Upload of model/EDA/MLflow artifacts.

Open GitHub -> Actions -> Heart Disease MLOps CI.

Take a screenshot showing a successful workflow.

The pipeline is deliberately configured to fail if linting or tests fail, satisfying the assignment's production-readiness requirement.

---

# 15. Stage 12 - Kubernetes deployment

The supplied `k8s/deployment.yaml` creates:

- 2 API replicas.
- Readiness probe.
- Liveness probe.
- CPU/memory requests and limits.
- LoadBalancer service.

## Option A: Docker Desktop Kubernetes

1. Enable Kubernetes in Docker Desktop.
2. Build the image:

```powershell
docker build -t heart-disease-api:latest .
```

3. Apply:

```powershell
kubectl apply -f k8s/deployment.yaml
```

4. Verify:

```powershell
kubectl get pods
kubectl get service heart-disease-api
```

5. Test:

```text
http://localhost/health
```

If your Kubernetes setup assigns a different external address, use the address shown by `kubectl get service`.

## Option B: Minikube

```powershell
minikube start
minikube image build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml
kubectl get pods
minikube service heart-disease-api --url
```

Open the returned URL and append `/docs` or `/health`.

For a real cloud deployment, push the image to Docker Hub, GitHub Container Registry, Azure Container Registry, ECR, or another registry and change the Kubernetes `image:` value.

Take screenshots of:

- `kubectl get pods`
- `kubectl get service`
- Successful `/health`
- Successful `/predict`

---

# 16. Stage 13 - Monitoring and logging

The API provides two lightweight monitoring mechanisms.

### Application logs

Every prediction request logs the prediction and confidence. Docker/Kubernetes can collect these stdout logs.

Docker:

```powershell
docker logs CONTAINER_ID
```

Kubernetes:

```powershell
kubectl logs deployment/heart-disease-api
```

### Prometheus metrics

The FastAPI application exposes `/metrics` using `prometheus-fastapi-instrumentator`.

Open:

```text
http://127.0.0.1:8000/metrics
```

For the assignment demonstration, generate several `/health` and `/predict` requests and show that request metrics are changing.

A full Prometheus/Grafana stack is intentionally not required for the basic implementation because the assignment allows API metrics/logs dashboard.

---

# 17. What screenshots you should capture

Save your own screenshots under `screenshots/` with these names:

```text
01_dataset_download.png
02_eda_class_balance.png
03_eda_correlation.png
04_mlflow_runs.png
05_mlflow_metrics.png
06_pytest.png
07_docker_build.png
08_docker_api.png
09_github_actions.png
10_kubernetes_pods.png
11_kubernetes_service.png
12_monitoring_metrics.png
```

Do not create fake screenshots. Capture them from your own execution.

---

# 18. Video recording plan

Record a 4-7 minute screen recording in this order:

1. Show project folder and README.
2. Show dataset download.
3. Show one or two EDA plots.
4. Run training and show model metrics.
5. Open MLflow and show runs.
6. Run pytest.
7. Build Docker image.
8. Run API and open Swagger.
9. Send `/predict` request.
10. Show GitHub Actions successful run.
11. Show Kubernetes pods/service.
12. Show `/metrics` and API logs.

Use the report screenshots as visual evidence.

---

# 19. How the solution maps to the marks

| Assignment task | Project evidence |
|---|---|
| 1. Data + EDA | `src/data.py`, `src/eda.py`, `reports/figures/` |
| 2. Feature engineering + models | `src/preprocess.py`, `src/train.py` |
| 3. Experiment tracking | MLflow in `src/train.py`, `mlruns/` |
| 4. Packaging | `models/*.joblib`, `requirements.txt`, preprocessing pipeline |
| 5. CI/CD + testing | `tests/`, `.github/workflows/ci.yml` |
| 6. Containerization | `Dockerfile`, `docker-compose.yml`, `src/api.py` |
| 7. Production deployment | `k8s/deployment.yaml` |
| 8. Monitoring/logging | FastAPI logs + `/metrics` |
| 9. Documentation | `REPORT.docx`, `REPORT.pdf`, `README.md` |

---

# 20. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root, for example:

```powershell
cd path\to\heart_disease_mlops
python -m src.data
```

### Dataset download fails

Use the manual-download fallback described in `data/README.md`.

### Docker says the model file is missing

Train first:

```powershell
python -m src.train
```

Then rebuild:

```powershell
docker build -t heart-disease-api:latest .
```

### Port 8000 is already in use

Run Docker with another host port:

```powershell
docker run --rm -p 8001:8000 heart-disease-api:latest
```

Then open `http://127.0.0.1:8001/docs`.

### Kubernetes image cannot be pulled

For Minikube, use:

```powershell
minikube image build -t heart-disease-api:latest .
```

For a real cluster, push the image to a registry and change the manifest image name.

---

# 21. Final submission checklist

Before submitting, confirm:

- [ ] Dataset download/cleaning works.
- [ ] EDA plots exist.
- [ ] Two models were trained.
- [ ] Cross-validation/tuning was executed.
- [ ] Accuracy, precision, recall, ROC-AUC were recorded.
- [ ] MLflow has all runs and artifacts.
- [ ] Model file exists.
- [ ] `pytest -q` passes.
- [ ] Flake8 passes.
- [ ] Docker image builds.
- [ ] Docker API works.
- [ ] Kubernetes deployment works.
- [ ] API logs are visible.
- [ ] `/metrics` works.
- [ ] GitHub Actions is green.
- [ ] Screenshots are real and from your execution.
- [ ] Report contains your actual metrics and screenshots.
- [ ] GitHub repository URL is added to the report.
- [ ] Video recording demonstrates the complete pipeline.

