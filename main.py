import io
import os
from datetime import datetime

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# =========================
# App Info (ثابتة للتقرير والواجهة)
# =========================
UNIVERSITY = "جامعة الكفيل"
COLLEGE = "كلية التقنيات الصحية والطبية"
DEPARTMENT = "قسم الأشعة"
STAGE = "المرحلة الرابعة"
PROJECT_TITLE = "COVID X-ray Classification (MobileNetV2)"

CLASS_NAMES = ["COVID", "Normal"]  # نفس ترتيب التدريب عندك
IMG_SIZE = (224, 224)

WEIGHTS_PATH = "covid.weights.h5"  # تأكد هذا الاسم مطابق في GitHub


app = FastAPI(title="COVID X-ray API (MobileNetV2)", version="1.0.0")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def build_model():
    # نفس المعمارية (بدون augmentation عند inference)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1.0 / 255.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    # compile غير ضروري للاستدلال
    return model


# تحميل الموديل مرة واحدة عند تشغيل السيرفر
MODEL = build_model()
if os.path.exists(WEIGHTS_PATH):
    MODEL.load_weights(WEIGHTS_PATH)
else:
    print(f"⚠️ Weights not found: {WEIGHTS_PATH}")


def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return img, x


def predict_from_bytes(file_bytes: bytes, threshold: float = 0.5):
    original_img, x = preprocess_image(file_bytes)

    prob_label1 = float(MODEL.predict(x, verbose=0)[0][0])  # P(label=1)=CLASS_NAMES[1]
    pred_label = 1 if prob_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]
    confidence = (prob_label1 if pred_label == 1 else (1 - prob_label1)) * 100.0

    return {
        "prob_label1": prob_label1,
        "threshold": threshold,
        "prediction": pred_name,
        "confidence": confidence,
        "label0": CLASS_NAMES[0],
        "label1": CLASS_NAMES[1],
    }, original_img


def make_pdf_report(result: dict, original_img: Image.Image):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    logo_path = os.path.join("static", "logo.png")
    y = height - 2.0 * cm

    # Draw logo (left)
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 1.5 * cm, y - 1.2 * cm, width=2.2 * cm, height=2.2 * cm, mask='auto')

    # Title text
    c.setFont("Helvetica-Bold", 14)
    c.drawString(4.2 * cm, y, UNIVERSITY)
    c.setFont("Helvetica", 11)
    c.drawString(4.2 * cm, y - 0.7 * cm, f"{COLLEGE} — {DEPARTMENT}")
    c.drawString(4.2 * cm, y - 1.4 * cm, f"{STAGE} | {PROJECT_TITLE}")

    c.setLineWidth(1)
    c.line(1.5 * cm, y - 2.0 * cm, width - 1.5 * cm, y - 2.0 * cm)

    # Info block
    y2 = y - 3.0 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.5 * cm, y2, "نتيجة الفحص")
    c.setFont("Helvetica", 11)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(1.5 * cm, y2 - 0.8 * cm, f"التاريخ والوقت: {now_str}")
    c.drawString(1.5 * cm, y2 - 1.6 * cm, f"Prediction: {result['prediction']}")
    c.drawString(1.5 * cm, y2 - 2.4 * cm, f"Confidence: {result['confidence']:.2f}%")
    c.drawString(1.5 * cm, y2 - 3.2 * cm, f"Raw sigmoid P(label=1): {result['prob_label1']:.4f}")
    c.drawString(1.5 * cm, y2 - 4.0 * cm, f"Threshold: {result['threshold']}")

    # Add X-ray image
    # نحفظ الصورة مؤقتًا داخل الذاكرة
    img_buf = io.BytesIO()
    original_img_rgb = original_img.convert("RGB")
    orig
