import os
import io
import numpy as np
import tensorflow as tf
import time
import logging

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.versioning import get_latest_version
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from src.performance_tracking import (
    log_prediction,
    get_performance_summary
)

# ==========================
# METRICS
# ==========================
REQUEST_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests"
)

logging.basicConfig(level=logging.INFO)

REQUEST_COUNT = 0

# ==========================
# APP & MODEL
# ==========================
app = FastAPI(title="Cats vs Dogs Inference API")

BASE_PATH = "models/prod"
if os.path.exists(BASE_PATH):
    prod_version = get_latest_version(BASE_PATH)
else:
    prod_version = None

MODEL_PATH = f"models/prod/{prod_version}/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# HELPERS
# ==========================
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ==========================
# ROUTES
# ==========================
@app.get("/health")
def health():
    return {"status": "healthy", "model_version": prod_version}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_label: str = None   # simulated ground truth
):
    global REQUEST_COUNT

    image_bytes = await file.read()

    start = time.time()
    with PREDICTION_LATENCY.time():
        image = preprocess(image_bytes)
        prob = model.predict(image)[0][0]

    latency = time.time() - start
    REQUEST_COUNT += 1

    label = "dog" if prob > 0.5 else "cat"

    # ==========================
    # PERFORMANCE TRACKING
    # ==========================
    if true_label is not None:
        log_prediction(true_label, label, float(prob))

    logging.info(
        f"Prediction | latency={latency:.3f}s | requests={REQUEST_COUNT}"
    )

    REQUEST_COUNTER.inc()

    return {
        "label": label,
        "confidence": float(prob),
        "model_version": prod_version,
        "latency": latency,
        "request_count": REQUEST_COUNT
    }


@app.get("/performance")
def performance():
    """
    Returns post-deployment model performance
    based on logged predictions.
    """
    return get_performance_summary()

@app.get("/metrics")
def metrics():
    """
    Exposes Prometheus metrics for scraping.
    """
    return Response(
        generate_latest(),
        media_type="text/plain; version=0.0.4"
    )
