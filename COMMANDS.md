# Assignment Commands

This file contains the commands used to execute and verify the MLOps Assignment-1 project.

## 1. Environment Setup

Create and activate the Python environment:

```bash
conda create -n mlopslab python=3.12.9
conda activate mlopslab
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data Acquisition

Download and inspect the dataset:

```bash
python -m src.data
```

This downloads the UCI Heart Disease dataset into:

```text
data/raw/heart_disease.csv
```

## 3. Exploratory Data Analysis

Generate the required EDA visualizations:

```bash
MPLBACKEND=Agg python -m src.eda
```

Generated figures are stored under:

```text
reports/figures/
```

## 4. Model Training

Start the MLflow tracking server:

```bash
mlflow server   --host 0.0.0.0   --port 5000   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root ./mlartifacts
```

In another terminal, activate the environment and run:

```bash
conda activate mlopslab
python -m src.train
```

The training process evaluates Logistic Regression and Random Forest models and saves the selected model under:

```text
models/
```

## 5. Reproducibility Check

Verify that the saved model can be loaded and used for prediction:

```bash
python -m src.predict
```

## 6. Unit Tests

Run the automated tests:

```bash
PYTHONPATH=. pytest -q
```

## 7. Linting

Run the code-quality check:

```bash
flake8 src tests --max-line-length=100
```

## 8. Docker Model Serving

Build the API image:

```bash
docker build -t heart-disease-api:v1 .
```

Run the container:

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:v1
```

Check the API:

```bash
curl http://localhost:8000/health
```

Test prediction:

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   --data @sample_request.json
```

Stop and remove the container when finished:

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

## 9. Kubernetes / Minikube Deployment

Start Minikube:

```bash
minikube start
```

Check the cluster:

```bash
minikube status
```

Build/load the application image for Minikube:

```bash
docker tag heart-disease-api:v1 heart-disease-api:latest
minikube image load heart-disease-api:latest
```

Deploy the application:

```bash
minikube kubectl -- apply -f k8s/deployment.yaml
```

Check deployment and pods:

```bash
minikube kubectl -- get deployments
minikube kubectl -- get pods
minikube kubectl -- get services
```

Restart the deployment when a locally loaded image has been updated:

```bash
minikube kubectl -- rollout restart deployment heart-disease-api
```

Get the service URL:

```bash
minikube service heart-disease-api --url
```

Verify the deployed API using the returned URL:

```bash
curl <SERVICE_URL>/health
curl <SERVICE_URL>/
```

Test prediction:

```bash
curl -X POST <SERVICE_URL>/predict   -H "Content-Type: application/json"   --data @sample_request.json
```

## 10. Prometheus Monitoring

Deploy Prometheus:

```bash
minikube kubectl -- apply -f k8s/monitoring/prometheus.yaml
```

Check Prometheus:

```bash
minikube kubectl -- get pods
minikube kubectl -- get services
```

Verify the API metrics endpoint:

```bash
curl <SERVICE_URL>/metrics
```

Check Prometheus targets from the Prometheus service when accessible:

```bash
curl http://127.0.0.1:9090/api/v1/targets
```

The Heart Disease API target should report:

```text
health: up
```

## 11. Grafana Monitoring

Deploy Grafana:

```bash
minikube kubectl -- apply -f k8s/monitoring/grafana.yaml
```

Check the service:

```bash
minikube kubectl -- get service grafana
```

Open Grafana using the NodePort/service URL provided by Minikube.

Configure the Prometheus datasource using:

```text
http://prometheus:9090
```

The monitoring dashboard should show API request count/rate, HTTP status information, and API latency.

## 12. GitHub Actions CI/CD

The workflow is located at:

```text
.github/workflows/ci.yml
```

It performs:

```text
Checkout
→ Python setup
→ Dependency installation
→ Linting
→ Dataset download
→ Unit tests
→ MLflow startup
→ Model training
→ Docker image build
→ Artifact upload
```

Push the project to the configured GitHub branch to trigger the workflow:

```bash
git add .
git commit -m "Update MLOps assignment"
git push
```

## 13. Git Status Verification

Before committing changes:

```bash
git status --short
```

Check staged changes:

```bash
git diff --cached --name-status
```

Check whitespace errors:

```bash
git diff --check
```

## Notes

- Run commands from the project root directory.
- Activate the project environment before running Python commands.
- `requirements.txt` contains Python dependencies; assignment execution commands are documented here.
- Replace `<SERVICE_URL>` with the URL returned by Minikube. Do not commit environment-specific URLs into source code.
