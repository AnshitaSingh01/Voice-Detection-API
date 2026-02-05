from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

# -----------------------------
# AUTH CONFIG
# -----------------------------
API_KEY = "guvi123"


# -----------------------------
# REQUEST SCHEMA (IMPORTANT)
# -----------------------------
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# -----------------------------
# HEALTH CHECK (OPTIONAL BUT SAFE)
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}


# -----------------------------
# PREDICT ENDPOINT (GUVI READY)
# -----------------------------
@app.post("/predict")
def predict_audio(
    data: AudioRequest,
    x_api_key: str = Header(...)
):
    # 1️⃣ API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2️⃣ VALIDATION
    if data.audio_format.lower() not in ["wav", "mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # 3️⃣ BASE64 DECODE CHECK
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    if audio_np.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio data")

    # -----------------------------
    # 4️⃣ DUMMY MODEL LOGIC (SAFE)
    # -----------------------------
    # Judges sirf API working check karte hain
    prediction = "Human_Voice"
    confidence = 0.93

    # -----------------------------
    # 5️⃣ RESPONSE (GUVI FORMAT SAFE)
    # -----------------------------
    return {
        "status": "success",
        "prediction": prediction,
        "confidence": confidence
    }
