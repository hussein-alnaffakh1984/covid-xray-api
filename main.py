from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

APP_TITLE = "COVID X-ray API (MobileNetV2)"
IMG_SIZE = (224, 224)

# نفس ترتيب التدريب غالبًا يطلع: ['COVID', 'Normal']
CLASS_NAMES = ["COVID", "Normal"]

app = FastAPI(title=APP_TITLE)

@tf.keras.utils.register_keras_serializable()
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

    # Build once so load_weights works reliably
    m(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    return m

MODEL = build_model()
MODEL.load_weights("covid.weights.h5")  # نفس اسم الملف داخل الريبو

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # لا /255 لأن Rescaling داخل النموذج
    return x

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "class_names": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    image_bytes = await file.read()
    x = preprocess(image_bytes)

    p_label1 = float(MODEL.predict(x, verbose=0)[0][0])  # P(CLASS_NAMES[1])
    pred_label = 1 if p_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]
    confidence = (p_label1 if pred_label == 1 else (1 - p_label1)) * 100

    return JSONResponse({
        "raw_sigmoid_p_label1": p_label1,
        "threshold": threshold,
        "prediction": pred_name,
        "confidence_percent": confidence
    })
