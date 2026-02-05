from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import librosa
import numpy as np
import base64
import requests
import uuid
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


app = FastAPI(
    title=" Voice Detection API",
    version="1.0"
)

# ================= CONFIG =================
API_KEY = "guvi123"
model = tf.keras.models.load_model("cnn_model.h5")

# ================= REQUEST SCHEMA =================
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str | None = None
    audio_url: str | None = None

# ================= PREDICT ENDPOINT =================
@app.post("/predict")
async def predict_audio(
    data: AudioRequest,
    x_api_key: str = Header(None)
):
    # 🔐 Authentication
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Validation: at least one audio input
    if not data.audio_base64 and not data.audio_url:
        raise HTTPException(
            status_code=400,
            detail="Either audio_base64 or audio_url must be provided"
        )

    temp_file = f"temp_{uuid.uuid4()}.{data.audio_format}"

    try:
        # ===== CASE 1: Base64 audio =====
        if data.audio_base64:
            audio_bytes = base64.b64decode(data.audio_base64)
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)

        # ===== CASE 2: Audio URL =====
        elif data.audio_url:
            r = requests.get(data.audio_url, timeout=10)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail="Audio download failed")
            with open(temp_file, "wb") as f:
                f.write(r.content)

        # ===== Audio Processing =====
        y, sr = librosa.load(temp_file, sr=22050)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = mfcc[:20, :20]
        mfcc = mfcc.reshape(1, 20, 20, 1)

        # ===== CNN Prediction =====
        pred = model.predict(mfcc)[0]
        label = "AI_Generated" if np.argmax(pred) == 1 else "Human_Voice"

        return {
            "status": "success",
            "prediction": label,
            "confidence": float(np.max(pred))
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Audio processing failed")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ================= HEALTH CHECK =================
@app.get("/")
def health():
    return {"message": "API is live and stable"}
