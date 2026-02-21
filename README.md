# Binary Image Classification – MLOps Pipeline

This project implements an end-to-end MLOps pipeline for a Cats vs Dogs
binary image classification use case.

## Tech Stack
- TensorFlow / Keras
- MLflow (Experiment Tracking)
- DVC (Data Versioning)
- FastAPI (Inference)
- Docker
- GitHub Actions (CI/CD)

## Project Structure
- data/ : raw and processed datasets
- src/ : training and inference code
- models/ : trained model artifacts
- tests/ : unit tests
- docker-compose.yml : local deployment

## Workflow
1. Data preprocessing and versioning
2. Model training and experiment tracking
3. Containerized inference service
4. CI/CD pipeline
5. Deployment and monitoring
