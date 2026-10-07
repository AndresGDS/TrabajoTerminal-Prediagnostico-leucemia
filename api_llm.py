r"""
api_llm.py

Backend del sistema HemVLM: BLIP genera la descripcion morfologica en
ingles, una deteccion deterministica por tipo de celula determina el
diagnostico (sana/leucemica), y un LLM pequeno (Qwen2.5-1.5B-Instruct)
traduce y normaliza la descripcion al espanol.

Uso (PowerShell), en una terminal:
    uvicorn api_llm:app --reload --port 8000

Modelos de descripcion seleccionables (GET /models, y campo opcional "modelo"
en POST /analyze): "vitb16" (por defecto, se carga al arrancar) y "fastvit"
(se carga la primera vez que se elige; requiere el checkpoint de
train_hemblip_fastvit.py y la libreria timm). Si no se envia "modelo", el
comportamiento es el mismo de siempre.

Requiere: el modelo Qwen2.5-1.5B-Instruct se descarga solo la primera vez
(~3GB). Si tu GPU va muy justa de memoria junto con BLIP, cambia
LLM_MODEL_NAME a "Qwen/Qwen2.5-0.5B-Instruct" (mucho mas ligero).
"""

import io
import os
import re
import threading
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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

# Variante con encoder FastViT (generada por train_hemblip_fastvit.py /
# fix_and_export_fastvit_model.py). Se puede cambiar la ruta con variables de
# entorno sin tocar el codigo.
FASTVIT_CHECKPOINT = os.environ.get(
    "HEMVLM_FASTVIT_CHECKPOINT", "./checkpoints/hemblip_fastvit/hemblip_fastvit_final.pt")
FASTVIT_PROCESSOR_DIR = os.environ.get(
    "HEMVLM_FASTVIT_PROCESSOR", "./checkpoints/hemblip_fastvit/processor")

# --------------------------------------------------------------------------- #
# Registro de modelos de descripcion. Solo hay entradas para modelos que
# realmente existen en el proyecto; el frontend las consulta en GET /models.
#   - "vitb16": se carga al arrancar (comportamiento original).
#   - "fastvit": se carga la primera vez que alguien lo elige (carga diferida),
#     porque con 4 GB de VRAM junto con BLIP + Qwen no siempre cabe en GPU.
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_KEY = "vitb16"
MODEL_REGISTRY = {
    "vitb16": {
        "etiqueta": "ViT-B/16",
        "descripcion": "Línea base del proyecto.",
        "tipo": "blip",
        "version": MODEL_VERSION_LABEL,
    },
    "fastvit": {
        "etiqueta": "FastViT",
        "descripcion": "Encoder alternativo FastViT, más ligero. Variante experimental del proyecto.",
        "tipo": "fastvit",
        "version": "HemVLM con encoder FastViT (BLIP + LoRA cross-attention)",
    },
}

app = FastAPI(title="HemVLM API (con post-procesamiento LLM)", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}            # clave -> dict(model, processor, preprocess, device, tipo)
_load_lock = threading.Lock()
llm_tokenizer = None
llm_model = None


def _load_vitb16() -> dict:
    """Modelo original: se carga exactamente como antes (from_pretrained)."""
    print(f"[HemVLM API] Cargando modelo de descripcion desde: {MODEL_DIR}")
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return {"model": model, "processor": processor, "preprocess": None,
            "device": device, "tipo": "blip"}


def _load_fastvit() -> dict:
    """FastViT no es un BlipForConditionalGeneration estandar (tiene el
    vision_model reemplazado), asi que se reutiliza la misma reconstruccion
    que ya valida generate_caption_fastvit.py / compare_encoders.py.
    Se carga primero en CPU; si hay VRAM libre se mueve a GPU y, si no cabe,
    se queda en CPU (mas lento, pero funciona)."""
    import timm
    from generate_caption_fastvit import load_model as build_fastvit_model

    print(f"[HemVLM API] Cargando variante FastViT desde: {FASTVIT_CHECKPOINT}")
    model = build_fastvit_model(FASTVIT_CHECKPOINT, "cpu")
    target = device
    if target == "cuda":
        try:
            model.to("cuda")
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            model.to("cpu")
            torch.cuda.empty_cache()
            target = "cpu"
            print("[HemVLM API] Sin VRAM suficiente para FastViT: se usara CPU.")
    model.eval()

    processor = BlipProcessor.from_pretrained(FASTVIT_PROCESSOR_DIR)
    # Mismo preprocesamiento que en la evaluacion: transform propio de timm
    data_config = timm.data.resolve_model_data_config(model.vision_model.backbone)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return {"model": model, "processor": processor, "preprocess": transform,
            "device": target, "tipo": "fastvit"}


_LOADERS = {"vitb16": _load_vitb16, "fastvit": _load_fastvit}


def model_availability(key: str):
    """(disponible, motivo). Comprueba archivos y dependencias sin cargar nada."""
    if key == "vitb16":
        if os.path.isdir(MODEL_DIR):
            return True, None
        return False, f"No se encontró la carpeta del modelo ({MODEL_DIR})."
    if key == "fastvit":
        if not os.path.isfile(FASTVIT_CHECKPOINT):
            return False, f"No se encontró el checkpoint ({FASTVIT_CHECKPOINT})."
        if not os.path.isdir(FASTVIT_PROCESSOR_DIR):
            return False, f"No se encontró la carpeta del processor ({FASTVIT_PROCESSOR_DIR})."
        try:
            import timm  # noqa: F401
        except ImportError:
            return False, "Falta instalar timm (pip install timm)."
        return True, None
    return False, "Modelo desconocido."


def get_model(key: str) -> dict:
    """Devuelve el modelo ya cargado o lo carga la primera vez que se pide."""
    if key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo desconocido: '{key}'. Opciones: {', '.join(MODEL_REGISTRY)}.",
        )
    if key in loaded_models:
        return loaded_models[key]

    etiqueta = MODEL_REGISTRY[key]["etiqueta"]
    available, reason = model_availability(key)
    if not available:
        raise HTTPException(status_code=503, detail=f"El modelo {etiqueta} no está disponible: {reason}")

    with _load_lock:
        if key not in loaded_models:
            try:
                loaded_models[key] = _LOADERS[key]()
            except Exception as e:
                raise HTTPException(
                    status_code=503, detail=f"No se pudo cargar el modelo {etiqueta}: {e}")
    return loaded_models[key]


@app.on_event("startup")
def load_models():
    global llm_tokenizer, llm_model

    loaded_models[DEFAULT_MODEL_KEY] = _LOADERS[DEFAULT_MODEL_KEY]()

    print(f"[HemVLM API] Cargando LLM de post-procesamiento: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    llm_model.eval()

    print(f"[HemVLM API] Listo. Dispositivo: {device}")


def generate_caption_en(image: Image.Image, bundle: dict) -> str:
    """Genera el caption en ingles con el modelo indicado. Los parametros de
    generacion son los mismos que se usaron en la evaluacion."""
    target = bundle["device"]
    if bundle["tipo"] == "fastvit":
        pixel_values = bundle["preprocess"](image).unsqueeze(0).to(target)
    else:
        pixel_values = bundle["processor"](images=image, return_tensors="pt").to(target)["pixel_values"]
    with torch.no_grad():
        output_ids = bundle["model"].generate(
            pixel_values=pixel_values, max_length=64, num_beams=4,
            repetition_penalty=1.5, no_repeat_ngram_size=3,
        )
    return bundle["processor"].decode(output_ids[0], skip_special_tokens=True)


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
        "status": "ok" if DEFAULT_MODEL_KEY in loaded_models and llm_model is not None else "modelos no cargados",
        "device": device,
        "modelo": MODEL_VERSION_LABEL,
        "modelo_llm": LLM_MODEL_NAME,
        "modelos_cargados": list(loaded_models),
    }


@app.get("/models")
def list_models():
    """Modelos de descripcion que el usuario puede elegir. 'disponible' indica
    si los archivos y dependencias necesarios existen; 'cargado' si ya estan
    en memoria (los que no, tardan unos segundos la primera vez)."""
    models = []
    for key, info in MODEL_REGISTRY.items():
        available, reason = model_availability(key)
        models.append({
            "clave": key,
            "etiqueta": info["etiqueta"],
            "descripcion": info["descripcion"],
            "disponible": available,
            "motivo": reason,
            "cargado": key in loaded_models,
            "por_defecto": key == DEFAULT_MODEL_KEY,
        })
    return {"modelos": models, "por_defecto": DEFAULT_MODEL_KEY}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), modelo: str = Form(DEFAULT_MODEL_KEY)):
    """`modelo` es opcional: si no se envia, se usa el modelo original
    (ViT-B/16), asi que los clientes anteriores siguen funcionando igual."""
    if DEFAULT_MODEL_KEY not in loaded_models or llm_model is None:
        raise HTTPException(status_code=503, detail="Los modelos no se pudieron cargar.")

    model_key = (modelo or DEFAULT_MODEL_KEY).strip().lower()
    bundle = get_model(model_key)   # 400 si no existe, 503 si no esta disponible

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    caption_en = generate_caption_en(image, bundle)
    llm_result = postprocess_with_llm(caption_en)
    info = MODEL_REGISTRY[model_key]

    return {
        "estado": "exito",
        "descripcion_en": caption_en,
        "descripcion": llm_result["descripcion_es"],
        "diagnostico_detectado": llm_result["diagnostico"],
        "confianza": llm_result["confianza"],
        "modelo": info["version"],
        "modelo_clave": model_key,
        "modelo_etiqueta": info["etiqueta"],
        "dispositivo_modelo": bundle["device"],
        "modelo_llm": LLM_MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
