"""
CoffeeGuard inference API.

Endpoints:
    GET  /health          -> liveness probe
    POST /predict         -> upload an image, get disease prediction
    GET  /metrics/summary -> basic served-prediction stats (for monitoring)
"""

import io
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

import pillow_heif
# Registers a PIL-compatible decoder for HEIC/HEIF — the default photo
# format on iPhones. Without this, any photo picked from an iOS photo
# library (not just the camera) fails Image.open() with a 400, even
# though the upload itself succeeded — this is the most common cause
# of an intermittent 400 from real phones.
pillow_heif.register_heif_opener()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import CLASS_NAMES, load_model_for_inference  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/coffeeguard.pt")
DB_PATH = os.getenv("PRED_LOG_DB", "monitoring/predictions.db")

CARE_TIPS = {
    "Healthy": "No disease detected. Keep up regular monitoring and good field hygiene.",
    "Miner": "Leaf miner detected. Remove and destroy affected leaves; consider approved biological or chemical control if infestation spreads.",
    "Phoma": "Phoma leaf spot detected. Improve field drainage and air circulation; avoid overhead irrigation; consider a copper-based fungicide.",
    "Red Spider Mite": "Red spider mite detected. Increase humidity around plants and consider miticide treatment if infestation is heavy.",
    "Rust": "Coffee leaf rust detected. Prune for airflow, remove infected leaves, and consider a recommended fungicide program.",
}

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            predicted_class TEXT,
            confidence REAL,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _init_db()
    if os.path.exists(MODEL_PATH):
        _model = load_model_for_inference(MODEL_PATH)
    else:
        # Allows the API/container to boot and pass health checks even
        # before a trained checkpoint is provided (e.g. in CI).
        _model = None
    yield


app = FastAPI(title="CoffeeGuard API", version="1.0.0", lifespan=lifespan)

# Needed so a browser-based client (or Flutter web build) can call this API
# from a different origin. Mobile (iOS/Android) builds aren't subject to
# CORS, but this is harmless and keeps the API usable everywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    start = time.time()
    raw = await file.read()

    # Don't trust the client-supplied content_type header — different
    # phones/OSes send inconsistent or missing values for it (e.g.
    # "image/jpg", empty, or "application/octet-stream"). Instead, try to
    # actually decode the bytes as an image; that's the real validation.
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read that file as an image. Please upload a JPEG or PNG.",
        )

    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])

    latency_ms = (time.time() - start) * 1000

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (ts, predicted_class, confidence, latency_ms) VALUES (?, ?, ?, ?)",
        (time.time(), CLASS_NAMES[pred_idx], confidence, latency_ms),
    )
    conn.commit()
    conn.close()

    predicted_class = CLASS_NAMES[pred_idx]
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "care_tip": CARE_TIPS.get(predicted_class, ""),
        "latency_ms": round(latency_ms, 2),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
    }


@app.get("/metrics/summary")
def metrics_summary():
    if not os.path.exists(DB_PATH):
        return {"total_predictions": 0}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT COUNT(*), AVG(latency_ms), AVG(confidence) FROM predictions")
    count, avg_latency, avg_conf = cur.fetchone()
    conn.close()
    return {
        "total_predictions": count or 0,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
        "avg_confidence": round(avg_conf, 4) if avg_conf else None,
    }