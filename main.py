import base64
import io
import os
from datetime import datetime

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request
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
CLASS_NAMES = ["COVID", "Normal"]  # label0, label1
IMG_SIZE = (224, 224)

WEIGHTS_PATH = "covid.weights.h5"
LOGO_PATH = "static/logo.png"

STAGE_TEXT = "Fourth Year"
ACADEMIC_YEAR_TEXT = "Academic Year 2025–2026"

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

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print(f"✅ Weights loaded: {WEIGHTS_PATH}")
else:
    print(f"❌ Weights file not found: {WEIGHTS_PATH}")


# =========================
# HELPERS
# =========================
def bytes_to_b64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def b64_to_bytes(b64_str: str) -> bytes:
    return base64.b64decode(b64_str.encode("utf-8"))


def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return img, x


def predict_bytes(file_bytes: bytes, threshold: float):
    pil_img, x = preprocess_image(file_bytes)
    prob_label1 = float(model.predict(x, verbose=0)[0][0])  # P(label=1) => CLASS_NAMES[1]

    pred_label = 1 if prob_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]

    # Confidence as the probability of the predicted class:
    conf = (prob_label1 if pred_label == 1 else (1.0 - prob_label1)) * 100.0

    result = {
        "prediction": pred_name,
        "confidence": float(conf),
        "prob_label1": float(prob_label1),
        "threshold": float(threshold),
        "label0": CLASS_NAMES[0],
        "label1": CLASS_NAMES[1],
    }
    return result, pil_img


def pil_to_data_uri(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def draw_pdf_header(c: canvas.Canvas, width: float, height: float):
    top_y = height - 60
    left_x = 40
    line_gap = 15

    c.setFont("Helvetica-Bold", 11)
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

    # Right side
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(width - 40, top_y, STAGE_TEXT)
    c.drawRightString(width - 40, top_y - 20, ACADEMIC_YEAR_TEXT)

    # Separator line
    c.setLineWidth(1)
    c.line(40, height - 150, width - 40, height - 150)


def build_pdf(result: dict, pil_img: Image.Image, patient_name: str, patient_age: str, examiner_name: str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    draw_pdf_header(c, width, height)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 190, "COVID-19 X-ray Classification Report")

    # Generated timestamp
    c.setFont("Helvetica", 11)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(40, height - 220, f"Generated: {ts}")

    # Patient/Examiner Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 245, "Patient / Examiner Information")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 265, f"Patient Name: {patient_name or '-'}")
    c.drawString(40, height - 285, f"Patient Age: {patient_age or '-'}")
    c.drawString(40, height - 305, f"Examiner Name: {examiner_name or '-'}")

    # Prediction block
    y = height - 340
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Prediction Result")

    c.setFont("Helvetica", 11)
    c.drawString(40, y - 20, f"Prediction: {result['prediction']}")
    c.drawString(40, y - 40, f"Confidence: {result['confidence']:.2f}%")
    c.drawString(40, y - 60, f"Raw Sigmoid P(label=1): {result['prob_label1']:.4f} (label=1 -> {result['label1']})")
    c.drawString(40, y - 80, f"Decision Threshold: {result['threshold']:.2f}")

    # Image (smaller and not covering text)
    img_buf = io.BytesIO()
    pil_img.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 115, "Analyzed X-ray Image")

    # Fixed safe area for image
    img_left = 60
    img_right = width - 60
    img_top = y - 135
    img_bottom = 95  # keep space for disclaimer

    img_w = img_right - img_left
    img_h = img_top - img_bottom

    # if very small, enforce a minimum (still safe)
    if img_h < 180:
        img_h = 180

    c.drawImage(
        ImageReader(img_buf),
        img_left,
        img_bottom,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto"
    )

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 60, "Disclaimer: This AI result is for educational/research use only and is not a medical diagnosis.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# =========================
# ROUTES
# =========================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "img_preview": None,
        "img_b64": None,
        "threshold": 0.5,
        "patient_name": "",
        "patient_age": "",
        "examiner_name": "",
    })


@app.post("/ui/predict", response_class=HTMLResponse, include_in_schema=False)
async def ui_predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    examiner_name: str = Form(""),
):
    file_bytes = await file.read()

    result, pil_img = predict_bytes(file_bytes, threshold)
    img_preview = pil_to_data_uri(pil_img)
    img_b64 = bytes_to_b64(file_bytes)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "img_preview": img_preview,
        "img_b64": img_b64,
        "threshold": threshold,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "examiner_name": examiner_name,
    })


@app.post("/ui/report", include_in_schema=False)
async def ui_report(
    img_b64: str = Form(...),
    threshold: float = Form(0.5),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    examiner_name: str = Form(""),
):
    file_bytes = b64_to_bytes(img_b64)
    result, pil_img = predict_bytes(file_bytes, threshold)

    pdf = build_pdf(result, pil_img, patient_name, patient_age, examiner_name)
    filename = f"covid_xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ---------- API endpoints (Swagger) ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "weights_found": os.path.exists(WEIGHTS_PATH),
        "logo_found": os.path.exists(LOGO_PATH),
        "class_names": CLASS_NAMES,
    }


@app.post("/predict")
async def api_predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    file_bytes = await file.read()
    result, _ = predict_bytes(file_bytes, threshold)
    return JSONResponse({
        **result,
        "confidence": round(result["confidence"], 2),
        "prob_label1": round(result["prob_label1"], 6),
    })


@app.post("/report")
async def api_report(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    examiner_name: str = Form(""),
):
    file_bytes = await file.read()
    result, pil_img = predict_bytes(file_bytes, threshold)

    pdf = build_pdf(result, pil_img, patient_name, patient_age, examiner_name)
    filename = f"covid_xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
