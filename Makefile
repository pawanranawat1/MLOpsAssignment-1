install:
	python -m pip install -r requirements.txt

data:
	python -m src.data

eda:
	python -m src.eda

train:
	python -m src.train

test:
	pytest -q

lint:
	flake8 src tests --max-line-length=100

api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest
