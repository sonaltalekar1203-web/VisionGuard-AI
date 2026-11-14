
import io
import os
import base64
import json
import sqlite3
from datetime import datetime

import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from anomalib.models import Patchcore


# --------------------------------------------------
# VISIONGUARD CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "/content/drive/MyDrive/VisionGuard/Models/visionguard_patchcore.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="VisionGuard AI",
    description="AI-powered industrial defect detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



# Load provisional classification threshold
THRESHOLD_PATH = "/content/VisionGuard-AI/backend/threshold.json"

with open(THRESHOLD_PATH, "r") as threshold_file:
    THRESHOLD_CONFIG = json.load(threshold_file)

VISIONGUARD_THRESHOLD = THRESHOLD_CONFIG["threshold"]



# --------------------------------------------------
# INSPECTION HISTORY DATABASE
# --------------------------------------------------

DATABASE_PATH = "/content/VisionGuard-AI/database/inspection_history.db"

def initialize_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
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

    connection.commit()
    connection.close()

initialize_database()


# --------------------------------------------------
# LOAD PATCHCORE MODEL
# --------------------------------------------------

print("Loading VisionGuard AI model...")

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

print("VisionGuard model loaded successfully!")
print("Running on:", DEVICE)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.ToTensor()


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "VisionGuard AI",
        "status": "Online",
        "device": DEVICE
    }


# --------------------------------------------------
# DEFECT DETECTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(image_tensor)

        score = float(prediction.pred_score[0].cpu())
        predicted_label = score > VISIONGUARD_THRESHOLD

        # Save inspection to SQLite database
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO inspections (
                filename, anomaly_score, prediction,
                inspected_at, image_width, image_height
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file.filename or "uploaded_image",
            score,
            "Defective" if predicted_label else "Normal",
            datetime.now().astimezone().isoformat(timespec="seconds"),
            image.width,
            image.height
        ))

        connection.commit()
        connection.close()

        anomaly_map = (
            prediction.anomaly_map[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # Normalize anomaly map for visualization
        heatmap = anomaly_map - anomaly_map.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_image = Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize(image.size)

        # Create a red heatmap overlay
        heatmap_array = np.array(heatmap_image)

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

        # Convert images to Base64
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "anomaly_score": score,
            "predicted_defect": predicted_label,
            "result": "Defective" if predicted_label else "Normal",
            "heatmap": image_to_base64(heatmap_image),
            "overlay": image_to_base64(overlay),
            "image_width": image.width,
            "image_height": image.height
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# INSPECTION HISTORY API
# --------------------------------------------------

@app.get("/history")
def get_inspection_history():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, filename, anomaly_score, prediction,
               inspected_at, image_width, image_height
        FROM inspections
        ORDER BY id DESC
    """)

    records = [dict(row) for row in cursor.fetchall()]
    connection.close()

    total = len(records)
    defective = sum(
        1 for record in records
        if record["prediction"] == "Defective"
    )
    normal = total - defective

    return {
        "total_inspections": total,
        "normal_products": normal,
        "defective_products": defective,
        "defect_percentage": (
            round(defective / total * 100, 2) if total else 0
        ),
        "records": records
    }

