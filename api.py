r"""
api.py

Backend FastAPI del sistema HemVLM. Carga el modelo entrenado UNA sola vez
al iniciar el servidor (no en cada request) y expone /analyze para generar
la descripcion morfologica de una imagen de celula.

Uso (PowerShell), en una terminal:
    uvicorn api:app --reload --port 8000

Documentacion interactiva automatica una vez corriendo:
    http://127.0.0.1:8000/docs
"""

import io
import re
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# --------------------------------------------------------------------------- #
# Configuracion -- ajusta si tu modelo final quedo en otra ruta
# --------------------------------------------------------------------------- #
MODEL_DIR = "./checkpoints/hemblip_exact_replica/hemblip_exact_replica_final"
MODEL_VERSION_LABEL = "HemVLM replica exacta (BLIP + LoRA cross-attention + ultimas 2 capas)"

DIAGNOSIS_KEYWORDS = {
    "acute myeloid leukemia": "AML -- Leucemia Mieloide Aguda",
    "aml": "AML -- Leucemia Mieloide Aguda",
    "acute lymphoblastic leukemia": "ALL -- Leucemia Linfoblastica Aguda",
    "all": "ALL -- Leucemia Linfoblastica Aguda",
}

app = FastAPI(title="HemVLM API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None


@app.on_event("startup")
def load_model():
    global processor, model
    try:
        processor = BlipProcessor.from_pretrained(MODEL_DIR)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        print(f"[HemVLM API] Modelo cargado en '{device}' desde: {MODEL_DIR}")
    except Exception as e:
        print(f"[HemVLM API] ERROR cargando el modelo: {e}")


def extract_diagnosis(caption_text: str) -> str:
    lower = caption_text.lower()
    for keyword, label in DIAGNOSIS_KEYWORDS.items():
        # \b = limite de palabra: evita que "all" coincida dentro de "small"
        # o "overall", y que "aml" coincida dentro de alguna otra palabra.
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, lower):
            return label
    return "Sin hallazgos de leucemia aguda (AML/ALL) en la descripcion generada"


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "modelo no cargado",
        "device": device,
        "modelo": MODEL_VERSION_LABEL,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="El modelo no se pudo cargar. Revisa la ruta MODEL_DIR.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=64,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    diagnosis = extract_diagnosis(caption)

    return {
        "estado": "exito",
        "descripcion": caption,
        "diagnostico_detectado": diagnosis,
        "modelo": MODEL_VERSION_LABEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
