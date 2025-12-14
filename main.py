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
    original_img_rgb.save(img_buf, format="PNG")
    img_buf.seek(0)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.5 * cm, y2 - 5.2 * cm, "صورة الأشعة المرفوعة")

    # Place image box
    img_x = 1.5 * cm
    img_y = 2.5 * cm
    img_w = width - 3.0 * cm
    img_h = (y2 - 6.0 * cm) - img_y

    c.drawImage(ImageReader(img_buf), img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(1.5 * cm, 1.5 * cm, "ملاحظة: هذا النموذج تعليمي/بحث تخرج ولا يُستخدم كتشخيص طبي نهائي.")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer


# لازم استيراد ImageReader بعد ما نستخدمه
from reportlab.lib.utils import ImageReader


@app.get("/", response_class=HTMLResponse)
def home():
    # صفحة بسيطة بدل تفاصيل /docs
    html = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>COVID X-ray</title>
  <style>
    body{font-family:Arial, sans-serif; background:#f6f7fb; margin:0; padding:0;}
    .wrap{max-width:900px; margin:24px auto; padding:16px;}
    .card{background:#fff; border-radius:14px; padding:18px; box-shadow:0 6px 20px rgba(0,0,0,.06);}
    .header{display:flex; align-items:center; gap:12px; margin-bottom:12px;}
    .logo{width:64px; height:64px; object-fit:contain;}
    .title h1{margin:0; font-size:20px;}
    .title p{margin:4px 0 0; color:#555;}
    .grid{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
    @media (max-width:820px){ .grid{grid-template-columns:1fr;} }
    .box{border:1px dashed #bbb; border-radius:12px; padding:12px; background:#fafafa;}
    .btn{display:inline-block; padding:10px 14px; border-radius:10px; border:0; cursor:pointer; font-weight:700;}
    .btn-primary{background:#2b6cb0; color:#fff;}
    .btn-secondary{background:#111; color:#fff;}
    .muted{color:#666; font-size:13px;}
    img.preview{width:100%; max-height:380px; object-fit:contain; border-radius:10px; background:#fff;}
    .result{margin-top:10px; padding:10px; background:#f1f5ff; border-radius:10px;}
    .row{display:flex; gap:10px; flex-wrap:wrap; align-items:center;}
    input[type="file"]{width:100%;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <img class="logo" src="/static/logo.png" alt="logo"/>
        <div class="title">
          <h1>جامعة الكفيل — كلية التقنيات الصحية والطبية</h1>
          <p>قسم الأشعة | المرحلة الرابعة — COVID X-ray Classification (MobileNetV2)</p>
        </div>
      </div>

      <div class="grid">
        <div class="box">
          <h3 style="margin-top:0;">رفع صورة الأشعة</h3>
          <input id="file" type="file" accept="image/*"/>
          <div class="muted" style="margin-top:8px;">ارفع JPG/PNG ثم اضغط “تشخيص”.</div>

          <div style="margin-top:12px;" class="row">
            <label>Threshold:</label>
            <input id="thr" type="number" min="0.1" max="0.9" step="0.05" value="0.5"/>
            <button class="btn btn-primary" onclick="runPredict()">تشخيص</button>
            <button class="btn btn-secondary" onclick="downloadPDF()" id="pdfBtn" disabled>تحميل تقرير PDF</button>
          </div>

          <div id="out" class="result" style="display:none;"></div>
        </div>

        <div class="box">
          <h3 style="margin-top:0;">معاينة الصورة</h3>
          <img id="preview" class="preview" src="" alt="preview" style="display:none;"/>
          <div id="noimg" class="muted">لم يتم رفع صورة بعد.</div>
        </div>
      </div>

      <div class="muted" style="margin-top:14px;">
        Swagger للتجربة التقنية: <a href="/docs" target="_blank">/docs</a>
      </div>
    </div>
  </div>

<script>
let lastFile = null;
document.getElementById("file").addEventListener("change", (e)=>{
  const f = e.target.files?.[0];
  lastFile = f || null;
  const img = document.getElementById("preview");
  const noimg = document.getElementById("noimg");
  if(!lastFile){ img.style.display="none"; noimg.style.display="block"; return; }
  img.src = URL.createObjectURL(lastFile);
  img.style.display="block";
  noimg.style.display="none";
  document.getElementById("out").style.display="none";
  document.getElementById("pdfBtn").disabled = true;
});

async function runPredict(){
  if(!lastFile){ alert("ارفع صورة أولاً"); return; }
  const thr = parseFloat(document.getElementById("thr").value || "0.5");
  const fd = new FormData();
  fd.append("file", lastFile);
  fd.append("threshold", String(thr));

  const res = await fetch("/predict", { method:"POST", body: fd });
  const data = await res.json();

  const out = document.getElementById("out");
  out.style.display = "block";
  out.innerHTML = `
    <b>Prediction:</b> ${data.prediction}<br/>
    <b>Confidence:</b> ${data.confidence.toFixed(2)}%<br/>
    <b>Raw P(label=1):</b> ${data.prob_label1.toFixed(4)}<br/>
    <b>Threshold:</b> ${data.threshold}
  `;
  document.getElementById("pdfBtn").disabled = false;
}

async function downloadPDF(){
  if(!lastFile){ alert("ارفع صورة أولاً"); return; }
  const thr = parseFloat(document.getElementById("thr").value || "0.5");
  const fd = new FormData();
  fd.append("file", lastFile);
  fd.append("threshold", String(thr));

  const res = await fetch("/report", { method:"POST", body: fd });
  if(!res.ok){ alert("حدث خطأ أثناء توليد التقرير"); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "covid_xray_report.pdf";
  a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/health")
def health():
    ok = os.path.exists(WEIGHTS_PATH)
    return {"status": "ok", "weights_found": ok}


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    file_bytes = await file.read()
    result, _ = predict_from_bytes(file_bytes, threshold=threshold)
    return JSONResponse(result)


@app.post("/report")
async def report(file: UploadFile = File(...), threshold: float = 0.5):
    file_bytes = await file.read()
    result, original_img = predict_from_bytes(file_bytes, threshold=threshold)

    pdf_buf = make_pdf_report(result, original_img)

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=covid_xray_report.pdf"}
    )
