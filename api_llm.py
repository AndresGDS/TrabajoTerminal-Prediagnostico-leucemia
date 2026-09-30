r"""
api_llm.py

Backend del sistema HemVLM: BLIP genera la descripcion morfologica en
ingles, una deteccion deterministica por tipo de celula determina el
diagnostico (sana/leucemica), y un LLM pequeno (Qwen2.5-1.5B-Instruct)
traduce y normaliza la descripcion al espanol.

Uso (PowerShell), en una terminal:
    uvicorn api_llm:app --reload --port 8000

Requiere: el modelo Qwen2.5-1.5B-Instruct se descarga solo la primera vez
(~3GB). Si tu GPU va muy justa de memoria junto con BLIP, cambia
LLM_MODEL_NAME a "Qwen/Qwen2.5-0.5B-Instruct" (mucho mas ligero).
"""

import io
import re
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer,
)


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
MODEL_DIR = "./checkpoints/hemblip_exact_replica/hemblip_exact_replica_final"
MODEL_VERSION_LABEL = "HemVLM replica exacta (BLIP + LoRA cross-attention + ultimas 2 capas)"

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # cambia a "Qwen/Qwen2.5-0.5B-Instruct" si falta VRAM

app = FastAPI(title="HemVLM API (con post-procesamiento LLM)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
caption_model = None
llm_tokenizer = None
llm_model = None


@app.on_event("startup")
def load_models():
    global processor, caption_model, llm_tokenizer, llm_model

    print(f"[HemVLM API] Cargando modelo de descripcion desde: {MODEL_DIR}")
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    caption_model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    caption_model.eval()

    print(f"[HemVLM API] Cargando LLM de post-procesamiento: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    llm_model.eval()

    print(f"[HemVLM API] Listo. Dispositivo: {device}")


def generate_caption_en(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = caption_model.generate(
            pixel_values=inputs["pixel_values"], max_length=64, num_beams=4,
            repetition_penalty=1.5, no_repeat_ngram_size=3,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast", "abnormal promyelocyte", "promyelocyte"]


def detect_diagnosis_from_celltype(caption_en: str) -> str:
    """Determina sana/leucemica de forma deterministica segun el tipo de
    celula mencionado (blasto conocido = leucemica), sin depender del LLM.
    El dataset fiel describe el tipo de celula en el texto, no la palabra
    'leucemia' -- por eso esta deteccion es por tipo celular, no por
    diagnostico textual."""
    lower = caption_en.lower()
    for keyword in BLAST_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return "leucemica"
    return "sana"


PROMPT_TEMPLATE = """Eres un asistente que normaliza descripciones morfologicas de celulas sanguineas generadas por un modelo de vision. Recibes una descripcion en ingles, posiblemente con errores gramaticales menores o cortes de texto.

Descripcion en ingles: "{caption_en}"

REGLAS ESTRICTAS, muy importantes:
- NUNCA inventes ni agregues atributos, hallazgos o palabras que no esten explicitamente en la descripcion en ingles de arriba. Si algo no se menciona, simplemente no lo incluyas -- no lo completes ni lo supongas.
- Usa terminologia hematologica estandar en espanol: "citoplasma" (nunca "liquido celular"), "vacuolas" (nunca "microcitos" salvo que la palabra "microcyte" aparezca literalmente en el texto en ingles), "cromatina", "nucleolo", "basofilia".
- Traduce fielmente el significado, no reinterpretes la forma o el tamano (si dice "round", es "redonda", no "conica" ni otra forma).

Responde SOLO con estas dos lineas, sin nada mas, sin explicaciones adicionales:
DESCRIPCION_ES: <traduccion fiel y natural en espanol, en una sola oracion clara, SIN agregar ningun atributo que no este en el texto en ingles>
CONFIANZA: <"alta" si la descripcion en ingles es clara y completa, "baja" si esta cortada, repetida, o es ambigua>"""


def postprocess_with_llm(caption_en: str) -> dict:
    diagnostico = detect_diagnosis_from_celltype(caption_en)  # determinista, no depende del LLM

    prompt = PROMPT_TEMPLATE.format(caption_en=caption_en)
    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs, max_new_tokens=120, do_sample=False, temperature=None, top_p=None,
        )
    generated = llm_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    descripcion_es = generated.strip()
    confianza = "no determinado"

    for line in generated.splitlines():
        line = line.strip()
        if line.upper().startswith("DESCRIPCION_ES:"):
            descripcion_es = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIANZA:"):
            confianza = line.split(":", 1)[1].strip().lower()

    return {
        "diagnostico": diagnostico,
        "descripcion_es": descripcion_es,
        "confianza": confianza,
        "texto_llm_crudo": generated.strip(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if caption_model is not None and llm_model is not None else "modelos no cargados",
        "device": device,
        "modelo": MODEL_VERSION_LABEL,
        "modelo_llm": LLM_MODEL_NAME,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if caption_model is None or llm_model is None:
        raise HTTPException(status_code=503, detail="Los modelos no se pudieron cargar.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    caption_en = generate_caption_en(image)
    llm_result = postprocess_with_llm(caption_en)

    return {
        "estado": "exito",
        "descripcion_en": caption_en,
        "descripcion": llm_result["descripcion_es"],
        "diagnostico_detectado": llm_result["diagnostico"],
        "confianza": llm_result["confianza"],
        "modelo": MODEL_VERSION_LABEL,
        "modelo_llm": LLM_MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
