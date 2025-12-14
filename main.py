import io
import os
from datetime import datetime

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# CONFIG (Edit if needed)
# =========================
CLASS_NAMES = ["COVID", "Normal"]       # Must match training order
IMG_SIZE = (224, 224)

WEIGHTS_PATH = "covid.weights.h5"       # <-- ensure this file exists in repo root
LOGO_PATH = "static/logo.png"           # <-- ensure exists

STAGE_TEXT = "Stage: Fourth Year"
ACADEMIC_YEAR_TEXT = "Academic Year: 2025–2026"

LEFT_HEADER_LINES = [
    "Republic of Iraq",
    "Ministry of Higher Education",
    "and Scientific Research",
    "University of Alkafeel",
    "College of Health & Medical Technology",
    "Dept. of Radiology Techniques",
]


# =========================
# APP
# =========================
app = FastAPI(title="COVID X-ray API (MobileNetV2)", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =========================
# MODEL
# =========================
def build_model():
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1.0 / 255.0),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


model = build_model()

if not os.path.exists(WEIGHTS_PATH):
    print(f"❌ Weights file not found: {WEIGHTS_PATH}")
else:
    model.load_weights(WEIGHTS_PATH)
    print(f"✅ Weights loaded: {WEIGHTS_PATH}")


def preprocess_image(file_bytes: bytes):
    # Return both PIL (for preview/pdf) and numpy batch (for model)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    return img, x


def predict_bytes(file_bytes: bytes, threshold: float = 0.5):
    pil_img, x = preprocess_image(file_bytes)

    # P(label=1) = P(CLASS_NAMES[1])
    prob_label1 = float(model.predict(x, verbose=0)[0][0])

    pred_label = 1 if prob_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]
    confidence = (prob_label1 if pred_label == 1 else (1 - prob_label1)) * 100.0

    return {
        "prob_label1": prob_label1,
        "threshold": float(threshold),
        "prediction": pred_name,
        "confidence": confidence,
        "label0": CLASS_NAMES[0],
        "label1": CLASS_NAMES[1],
    }, pil_img


def pdf_header(c: canvas.Canvas, width: float, height: float):
    """
    Draw:
    - LEFT: official lines
    - CENTER: logo
    - RIGHT: stage + academic year
    """
    top_y = height - 60
    left_x = 40

    c.setFont("Helvetica-Bold", 11)
    line_gap = 15
    for i, line in enumerate(LEFT_HEADER_LINES):
        c.drawString(left_x, top_y - i * line_gap, line)

    # Center logo
    if os.path.exists(LOGO_PATH):
        c.drawImage(
            LOGO_PATH,
            (width / 2) - 40,
            height - 120,
            width=80,
            height=80,
            mask="auto"
        )

    # Right block
    right_x = width - 200
    c.drawString(right_x, top_y, STAGE_TEXT)
    c.drawString(right_x, top_y - 20, ACADEMIC_YEAR_TEXT)

    # Separator line
    c.setLineWidth(1)
    c.line(40, height - 150, width - 40, height - 150)


def build_pdf_report(result: dict, pil_img: Image.Image) -> io.BytesIO:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    width, height = A4

    # Header
    pdf_header(c, width, height)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 190, "COVID-19 X-ray Classification Report")

    # Timestamp
    c.setFont("Helvetica", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(40, height - 220, f"Generated: {ts}")

    # Results block
    y = height - 255
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Prediction Result")
    c.setFont("Helvetica", 11)
    c.drawString(40, y - 20, f"Prediction: {result['prediction']}")
    c.drawString(40, y - 40, f"Confidence: {result['confidence']:.2f}%")
    c.drawString(40, y - 60, f"Raw Sigmoid P(label=1): {result['prob_label1']:.4f}  (label=1 -> {result['label1']})")
    c.drawString(40, y - 80, f"Decision Threshold: {result['threshold']:.2f}")

    # Image section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 115, "Analyzed X-ray Image")

    # Convert PIL to buffer for ReportLab
    img_buf = io.BytesIO()
    pil_img.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)

    # Draw image
    img_x = 40
    img_y = 90
    img_w = width - 80
    img_h = (y - 135) - img_y
    if img_h < 200:
        img_h = 200

    c.drawImage(
        ImageReader(img_buf),
        img_x, img_y,
        width=img_w, height=img_h,
        preserveAspectRatio=True,
        anchor='c',
        mask="auto"
    )

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        40, 60,
        "Disclaimer: This AI result is for educational/research use only and must not be used as a standalone medical diagnosis."
    )

    c.showPage()
    c.save()
    buff.seek(0)
    return buff


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Requires templates/index.html (provided below).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "weights_found": os.path.exists(WEIGHTS_PATH),
        "logo_found": os.path.exists(LOGO_PATH),
        "class_names": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    file_bytes = await file.read()
    result, _ = predict_bytes(file_bytes, threshold=threshold)
    # round for clean UI
    return JSONResponse({
        **result,
        "confidence": round(result["confidence"], 2),
        "prob_label1": round(result["prob_label1"], 6),
    })


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    file_bytes = await file.read()
    result, pil_img = predict_bytes(file_bytes, threshold=threshold)

    pdf_buf = build_pdf_report(result, pil_img)
    filename = f"covid_xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
