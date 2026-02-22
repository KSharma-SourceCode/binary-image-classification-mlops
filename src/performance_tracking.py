import csv
import os
from datetime import datetime
import pandas as pd

LOG_FILE = "logs/predictions.csv"


def log_prediction(y_true, y_pred, confidence):
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                ["timestamp", "true_label", "predicted_label", "confidence"]
            )

        writer.writerow([
            datetime.now(),
            y_true,
            y_pred,
            confidence
        ])


# ================================
# PERFORMANCE SUMMARY
# ================================
def get_performance_summary():
    if not os.path.exists(LOG_FILE):
        return {
            "message": "No prediction data available yet"
        }

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        return {
            "message": "No predictions logged"
        }

    total = len(df)
    correct = (df["true_label"] == df["prediction"]).sum()
    accuracy = correct / total

    return {
        "total_predictions": int(total),
        "correct_predictions": int(correct),
        "accuracy": round(float(accuracy), 4),
        "avg_confidence": round(float(df["confidence"].mean()), 4)
    }