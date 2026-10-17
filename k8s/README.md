# Kubernetes deployment

## Docker Desktop Kubernetes

1. Build the image:
   `docker build -t heart-disease-api:latest .`
2. Make sure Kubernetes is enabled in Docker Desktop.
3. Apply the manifest:
   `kubectl apply -f k8s/deployment.yaml`
4. Check pods:
   `kubectl get pods`
5. Check service:
   `kubectl get service heart-disease-api`
6. On Docker Desktop, open:
   `http://localhost/health`

## Minikube

1. Start Minikube:
   `minikube start`
2. Build the image inside Minikube:
   `minikube image build -t heart-disease-api:latest .`
3. Apply:
   `kubectl apply -f k8s/deployment.yaml`
4. Check:
   `kubectl get pods`
5. Open the service:
   `minikube service heart-disease-api --url`

For a real cloud deployment, push the image to a container registry and replace the
`image:` value in `deployment.yaml` with the registry image name.
