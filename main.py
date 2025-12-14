import io
import os
import base64
import hashlib
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# CONFIG
# =========================
CLASS_NAMES = ["COVID", "Normal"]   # label0, label1
IMG_SIZE = (224, 224)

WEIGHTS_PATH = "covid.weights.h5"
LOGO_PATH = "static/logo.png"

STAGE_TEXT = "Fourth Year"
ACADEMIC_YEAR_TEXT = "Academic Year 2025-2026"

LEFT_HEADER_LINES = [
    "Republic of Iraq",
    "Ministry of Higher Education",
    "and Scientific Research",
    "University of Alkafeel",
    "College of Health & Medical Technology",
    "Dept. of Radiology Techniques",
]

# ---- Simple in-memory store for last diagnosis (single-user friendly) ----
LAST_STATE = {
    "img_bytes": None,          # type: Optional[bytes]
    "threshold": 0.5,           # type: float
    "patient_name": "",         # type: str
    "examiner_name": "",        # type: str
    "result": None,             # type: Optional[dict]
    "img_preview": None,        # type: Optional[str]
    "updated_at": None,         # type: Optional[str]
}


# =========================
# APP
# =========================
app = FastAPI(title="COVID X-ray API (MobileNetV2)", version="1.0.0")

if not os.path.isdir("static"):
    os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =========================
# MODEL
# =========================
def build_model():
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
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

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights loaded")
else:
    print(f"❌ Weights file not found: {WEIGHTS_PATH}")


# =========================
# HELPERS
# =========================
def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return img, x


def predict_bytes(file_bytes: bytes, threshold: float):
    pil_img, x = preprocess_image(file_bytes)
    prob = float(model.predict(x, verbose=0)[0][0])  # P(label=1) => Normal

    label = 1 if prob >= threshold else 0
    prediction = CLASS_NAMES[label]
    confidence = prob if label == 1 else (1 - prob)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "prob_label1": round(prob, 6),
        "threshold": float(threshold),
        "label0": CLASS_NAMES[0],
        "label1": CLASS_NAMES[1],
    }, pil_img


def pil_to_data_uri(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def draw_pdf_header(c, width, height):
    top_y = height - 60
    left_x = 40

    c.setFont("Helvetica-Bold", 11)
    for i, line in enumerate(LEFT_HEADER_LINES):
        c.drawString(left_x, top_y - i * 15, line)

    if os.path.exists(LOGO_PATH):
        c.drawImage(LOGO_PATH, width / 2 - 40, height - 120, 80, 80, mask="auto")

    c.drawRightString(width - 40, top_y, STAGE_TEXT)
    c.drawRightString(width - 40, top_y - 20, ACADEMIC_YEAR_TEXT)

    c.setLineWidth(1)
    c.line(40, height - 150, width - 40, height - 150)


def build_pdf(result: dict, pil_img: Image.Image, patient_name: str, examiner_name: str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    draw_pdf_header(c, width, height)

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 190, "COVID-19 X-ray Classification Report")

    c.setFont("Helvetica", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(40, height - 220, f"Generated: {ts}")

    # Patient / Examiner
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 245, "Patient / Examiner Information")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 265, f"Patient Name: {patient_name or '-'}")
    c.drawString(40, height - 283, f"Examiner Name: {examiner_name or '-'}")

    # Results
    y = height - 315
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Prediction Result")
    c.setFont("Helvetica", 11)
    c.drawString(40, y - 20, f"Prediction: {result['prediction']}")
    c.drawString(40, y - 40, f"Confidence: {result['confidence']:.2f}%")
    c.drawString(40, y - 60, f"Raw Sigmoid P(label=1): {result['prob_label1']:.6f} (label=1 -> {result['label1']})")
    c.drawString(40, y - 80, f"Threshold: {result['threshold']:.2f}")

    # Image (smaller)
    img_buf = io.BytesIO()
    pil_img.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)

    img_x = 80
    img_y = 110
    img_w = width - 160
    img_h = 240  # smaller to avoid overlapping text

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, img_y + img_h + 12, "Analyzed X-ray Image")

    c.drawImage(
        ImageReader(img_buf),
        img_x, img_y,
        width=img_w, height=img_h,
        preserveAspectRatio=True,
        anchor='c',
        mask="auto"
    )

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 80, "Disclaimer: Educational/research use only. Not a medical diagnosis.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": LAST_STATE["result"],
        "img_preview": LAST_STATE["img_preview"],
        "threshold": LAST_STATE["threshold"],
        "patient_name": LAST_STATE["patient_name"],
        "examiner_name": LAST_STATE["examiner_name"],
        "has_last": LAST_STATE["img_bytes"] is not None,
        "updated_at": LAST_STATE["updated_at"],
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "weights_found": os.path.exists(WEIGHTS_PATH),
        "logo_found": os.path.exists(LOGO_PATH),
    }


@app.post("/ui/predict", response_class=HTMLResponse)
async def ui_predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    patient_name: str = Form(""),
    examiner_name: str = Form(""),
):
    file_bytes = await file.read()
    result, pil_img = predict_bytes(file_bytes, threshold)
    img_preview = pil_to_data_uri(pil_img)

    LAST_STATE["img_bytes"] = file_bytes
    LAST_STATE["threshold"] = float(threshold)
    LAST_STATE["patient_name"] = patient_name.strip()
    LAST_STATE["examiner_name"] = examiner_name.strip()
    LAST_STATE["result"] = result
    LAST_STATE["img_preview"] = img_preview
    LAST_STATE["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "img_preview": img_preview,
        "threshold": threshold,
        "patient_name": patient_name,
        "examiner_name": examiner_name,
        "has_last": True,
        "updated_at": LAST_STATE["updated_at"],
    })


@app.post("/report_last")
async def report_last():
    """Generate PDF using the last diagnosed image (no re-upload)."""
    if LAST_STATE["img_bytes"] is None or LAST_STATE["result"] is None:
        raise HTTPException(status_code=400, detail="No diagnosis found yet. Please diagnose an image first.")

    file_bytes = LAST_STATE["img_bytes"]
    threshold = LAST_STATE["threshold"]
    patient_name = LAST_STATE["patient_name"]
    examiner_name = LAST_STATE["examiner_name"]

    # Use stored result if available, but rebuild image object for PDF
    result = LAST_STATE["result"]
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    pdf = build_pdf(result, pil_img, patient_name, examiner_name)
    filename = f"covid_xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )


@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    file_bytes = await file.read()
    result, _ = predict_bytes(file_bytes, threshold)
    return JSONResponse(result)
