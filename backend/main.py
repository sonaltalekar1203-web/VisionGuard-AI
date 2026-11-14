
import io
import os
import base64
import json
import sqlite3
import logging
import threading

from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

import torch
import numpy as np

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from anomalib.models import Patchcore


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "VISIONGUARD_MODEL_PATH",
        "/content/drive/MyDrive/VisionGuard/Models/visionguard_patchcore.pth"
    )
)

THRESHOLD_PATH = Path(
    os.getenv(
        "VISIONGUARD_THRESHOLD_PATH",
        str(BACKEND_DIR / "threshold.json")
    )
)

DATABASE_PATH = Path(
    os.getenv(
        "VISIONGUARD_DATABASE_PATH",
        str(PROJECT_DIR / "database" / "inspection_history.db")
    )
)

MAX_UPLOAD_SIZE = int(
    os.getenv("VISIONGUARD_MAX_UPLOAD_MB", "10")
) * 1024 * 1024

MAX_IMAGE_PIXELS = int(
    os.getenv("VISIONGUARD_MAX_IMAGE_PIXELS", "25000000")
)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INFERENCE_LOCK = threading.Lock()


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("VisionGuard")


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="VisionGuard AI",
    description=(
        "AI-powered industrial anomaly detection "
        "and predictive quality intelligence API."
    ),
    version="2.0.0"
)

allowed_origins = os.getenv(
    "VISIONGUARD_CORS_ORIGINS",
    "http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)


# ============================================================
# 4. LOAD CLASSIFICATION THRESHOLD
# ============================================================

if not THRESHOLD_PATH.is_file():
    raise FileNotFoundError(
        f"Threshold configuration not found: {THRESHOLD_PATH}"
    )

with THRESHOLD_PATH.open("r", encoding="utf-8") as threshold_file:
    THRESHOLD_CONFIG = json.load(threshold_file)

VISIONGUARD_THRESHOLD = float(THRESHOLD_CONFIG["threshold"])

if not np.isfinite(VISIONGUARD_THRESHOLD):
    raise ValueError("Invalid anomaly threshold configuration.")

logger.info(
    "Classification threshold loaded: %.4f",
    VISIONGUARD_THRESHOLD
)


# ============================================================
# 5. DATABASE MANAGEMENT
# ============================================================

@contextmanager
def get_database_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_database_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                prediction TEXT NOT NULL,
                inspected_at TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER
            )
        """)

    logger.info("Inspection database initialized.")


initialize_database()


# ============================================================
# 6. LOAD PATCHCORE MODEL
# ============================================================

logger.info("Loading VisionGuard PatchCore model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"PatchCore checkpoint not found: {MODEL_PATH}. "
        "Set VISIONGUARD_MODEL_PATH to the correct checkpoint location."
    )

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9
)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)
model.eval()
model.to(DEVICE)

logger.info("VisionGuard model loaded successfully.")
logger.info("Inference device: %s", DEVICE)


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

transform = transforms.ToTensor()


def load_and_validate_image(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Supported image formats: JPEG, PNG and WEBP."
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the allowed limit."
                )

            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )


# ============================================================
# 8. IMAGE ENCODING
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# 9. ANOMALY MAP VISUALIZATION
# ============================================================

def generate_visualizations(image, anomaly_map):

    heatmap = np.asarray(anomaly_map, dtype=np.float32)

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    heatmap = heatmap - heatmap.min()

    maximum = heatmap.max()

    if maximum > 0:
        heatmap = heatmap / maximum

    heatmap_image = Image.fromarray(
        (heatmap * 255).clip(0, 255).astype(np.uint8)
    ).resize(image.size)

    heatmap_array = np.asarray(heatmap_image)

    red_overlay = np.zeros(
        (image.height, image.width, 3),
        dtype=np.uint8
    )

    red_overlay[:, :, 0] = heatmap_array

    overlay = Image.blend(
        image,
        Image.fromarray(red_overlay),
        alpha=0.45
    )

    return heatmap_image, overlay


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "project": "VisionGuard AI",
        "status": "Online",
        "version": app.version,
        "device": DEVICE,
        "model": "PatchCore",
        "threshold": VISIONGUARD_THRESHOLD
    }


# ============================================================
# 11. DEFECT DETECTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Maximum upload size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image = load_and_validate_image(image_bytes)

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Run inference safely
        with INFERENCE_LOCK:
            with torch.inference_mode():
                prediction = model(image_tensor)

        score = float(prediction.pred_score[0].detach().cpu())

        if not np.isfinite(score):
            raise RuntimeError("Model returned an invalid anomaly score.")

        predicted_label = score > VISIONGUARD_THRESHOLD

        result = (
            "Defective" if predicted_label else "Normal"
        )

        # Extract anomaly map
        anomaly_map = prediction.anomaly_map[0].detach().cpu().numpy()

        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[0]

        heatmap_image, overlay = generate_visualizations(
            image,
            anomaly_map
        )

        # Store inspection in database
        with get_database_connection() as connection:

            connection.execute("""
                INSERT INTO inspections (
                    filename,
                    anomaly_score,
                    prediction,
                    inspected_at,
                    image_width,
                    image_height
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file.filename or "uploaded_image",
                score,
                result,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                image.width,
                image.height
            ))

        logger.info(
            "Inspection completed | File: %s | Score: %.4f | Result: %s",
            file.filename,
            score,
            result
        )

        return {
            "filename": file.filename or "uploaded_image",
            "anomaly_score": score,
            "predicted_defect": predicted_label,
            "result": result,
            "heatmap": image_to_base64(heatmap_image),
            "overlay": image_to_base64(overlay),
            "image_width": image.width,
            "image_height": image.height
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Image inspection failed.")

        raise HTTPException(
            status_code=500,
            detail="Image inspection failed due to an internal server error."
        )

    finally:
        await file.close()


# ============================================================
# 12. INSPECTION HISTORY API
# ============================================================

@app.get("/history")
def get_inspection_history():

    try:
        with get_database_connection() as connection:

            records = connection.execute("""
                SELECT
                    id,
                    filename,
                    anomaly_score,
                    prediction,
                    inspected_at,
                    image_width,
                    image_height
                FROM inspections
                ORDER BY id DESC
            """).fetchall()

        records = [dict(record) for record in records]

        total = len(records)

        defective = sum(
            1 for record in records
            if record["prediction"] == "Defective"
        )

        normal = total - defective

        defect_percentage = (
            round((defective / total) * 100, 2)
            if total > 0 else 0
        )

        return {
            "total_inspections": total,
            "normal_products": normal,
            "defective_products": defective,
            "defect_percentage": defect_percentage,
            "records": records
        }

    except Exception:
        logger.exception("Failed to retrieve inspection history.")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve inspection history."
        )


# ============================================================
# 13. APPLICATION STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("VisionGuard AI API is ready.")
    logger.info("Running on device: %s", DEVICE)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("VisionGuard AI API shutting down.")
