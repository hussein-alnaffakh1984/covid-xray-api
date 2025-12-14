from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from datetime import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

APP_TITLE = "COVID X-ray API (MobileNetV2)"
IMG_SIZE = (224, 224)

# ترتيب التدريب غالبًا: ['COVID','Normal']
CLASS_NAMES = ["COVID", "Normal"]

# نجعل Swagger اختياري وباسم مختلف
app = FastAPI(title=APP_TITLE, docs_url="/api-docs", redoc_url=None)

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def build_model():
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ])

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    m = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_aug,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    # build once
    m(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    return m


MODEL = build_model()
MODEL.load_weights("covid.weights.h5")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # no /255 (Rescaling داخل النموذج)
    return x


def predict_bytes(image_bytes: bytes, threshold: float = 0.5):
    x = preprocess(image_bytes)
    p_label1 = float(MODEL.predict(x, verbose=0)[0][0])  # P(CLASS_NAMES[1])
    pred_label = 1 if p_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]
    confidence = (p_label1 if pred_label == 1 else (1 - p_label1)) * 100

    return {
        "raw_sigmoid_p_label1": p_label1,
        "threshold": threshold,
        "prediction": pred_name,
        "confidence_percent": confidence
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "class_names": CLASS_NAMES}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...), threshold: float = 0.5):
    image_bytes = await file.read()
    result = predict_bytes(image_bytes, float(threshold))
    return JSONResponse(result)


@app.post("/ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...), threshold: float = Form(0.5)):
    image_bytes = await file.read()
    thr = float(threshold)

    # Prediction
    result = predict_bytes(image_bytes, thr)

    # Show the same image (Base64)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    img_src = f"data:image/png;base64,{img_b64}"
    result["image_src"] = img_src

    return templates.TemplateResponse("index.html", {"request": request, "result": result})


@app.get("/report-page", response_class=HTMLResponse)
def report_page():
    return HTMLResponse("""
    <html lang="ar" dir="rtl">
    <head><meta charset="utf-8"><title>تقرير PDF</title></head>
    <body style="font-family:Arial; background:#f6f7fb; padding:20px;">
      <div style="max-width:700px;margin:auto;background:#fff;padding:18px;border-radius:12px;">
        <h2>تحميل تقرير PDF</h2>
        <p>ارفع نفس صورة الأشعة مرة أخرى لتوليد التقرير.</p>
        <form action="/report" method="post" enctype="multipart/form-data">
          <label>الصورة:</label><br>
          <input type="file" name="file" accept=".jpg,.jpeg,.png" required><br><br>
          <label>Threshold:</label><br>
          <input type="number" name="threshold" step="0.05" min="0.1" max="0.9" value="0.5"><br><br>
          <button type="submit" style="padding:10px 14px;border:0;border-radius:10px;background:#2f6fed;color:#fff;cursor:pointer;">
            توليد وتحميل PDF
          </button>
        </form>
        <p style="color:#666;margin-top:12px;">ملاحظة: لأغراض تعليمية/بحثية فقط.</p>
        <a href="/" style="display:inline-block;margin-top:10px;">رجوع للواجهة</a>
      </div>
    </body>
    </html>
    """)


@app.post("/report")
async def report_pdf(file: UploadFile = File(...), threshold: float = Form(0.5)):
    image_bytes = await file.read()
    thr = float(threshold)
    result = predict_bytes(image_bytes, thr)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 70, "COVID X-ray Classification Report (MobileNetV2)")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "University: Al-Kafeel University")
    c.drawString(50, height - 120, "College: College of Health & Medical Technologies")
    c.drawString(50, height - 140, "Department: Radiology")
    c.drawString(50, height - 160, "Stage: 4th Year")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 200, f"Prediction: {result['prediction']}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 220, f"Confidence: {result['confidence_percent']:.2f}%")
    c.drawString(50, height - 240, f"Threshold: {thr}")
    c.drawString(50, height - 260, f"Raw sigmoid P(label=1): {result['raw_sigmoid_p_label1']:.4f}")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 290, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 60, "Disclaimer: Educational/Research use only. Not a medical diagnosis.")

    c.showPage()
    c.save()
    buf.seek(0)

    filename = "covid_xray_report.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
