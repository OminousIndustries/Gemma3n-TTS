"""
FastAPI wrapper for your local Gemma-3n model.

• POST /ask         – audio→text (unchanged)
• POST /ask_image  – image+prompt→text (new)

CORS is open for http://localhost:5173 so the React front-end can call us.
"""

import base64, os, tempfile, torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gemma_record_gui import get_model_and_processor, sanitize

# --------------------------------------------------------------------
# FastAPI + CORS
# --------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Load model/processor once
# --------------------------------------------------------------------

model, processor = get_model_and_processor()

# --------------------------------------------------------------------
# /ask  — audio blob (base-64)  →  text
# --------------------------------------------------------------------

class AudioPayload(BaseModel):
    data: str                        # base-64 WAV data (no "data:…," prefix)

@app.post("/ask")
async def ask_audio(payload: AudioPayload):
    wav_path = None
    try:
        wav_bytes = base64.b64decode(payload.data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
            tmp.write(wav_bytes)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a friendly assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is my audio message:"},
                    {"type": "audio", "audio": wav_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, disable_compile=True)

        reply = processor.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return {"text": sanitize(reply)}

    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

# --------------------------------------------------------------------
# /ask_image  — multipart(form-data)  →  text
# --------------------------------------------------------------------

@app.post("/ask_image")
async def ask_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    img_path = None
    try:
        suffix = os.path.splitext(image.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            img_path = tmp.name
            tmp.write(await image.read())

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a friendly assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, disable_compile=True)

        reply = processor.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return {"text": sanitize(reply)}

    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
