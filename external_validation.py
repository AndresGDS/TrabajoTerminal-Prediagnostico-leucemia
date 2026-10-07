r"""
external_validation.py

Validacion externa real: corre el modelo sobre un dataset de leucemicas de
UNA FUENTE QUE NUNCA VIO durante el entrenamiento (ni WBCAtt ni
LeukemiaAttri), y calcula que porcentaje detecta correctamente como
"leucemica" -- la sensibilidad en dominio externo, analoga a la caida de
85% a 52% que reporta el paper original en su propia validacion externa.

Asume que TODAS las imagenes en --images_dir son casos leucemicos
verdaderos (no requiere ningun archivo de etiquetas).

Uso (PowerShell):
    python external_validation.py `
        --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
        --images_dir .\ruta\al\dataset_externo `
        --out_csv .\validacion_externa.csv
"""

import argparse
import os
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast", "abnormal promyelocyte", "promyelocyte"]


def detect_diagnosis_from_celltype(caption_en: str) -> str:
    lower = caption_en.lower()
    for keyword in BLAST_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return "leucemica"
    return "sana"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True,
                         help="Carpeta con las imagenes externas (se asume que TODAS son leucemicas)")
    parser.add_argument("--out_csv", type=str, default="./validacion_externa.csv")
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if f.lower().endswith(valid_ext)
    ]
    print(f"Imagenes encontradas: {len(image_files)}")

    rows = []
    n_correct = 0
    for path in tqdm(image_files, desc="Evaluando en dominio externo"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=inputs["pixel_values"], max_length=args.max_length, num_beams=4,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        prediccion = detect_diagnosis_from_celltype(caption)
        acierto = prediccion == "leucemica"
        n_correct += int(acierto)

        rows.append({
            "image_path": path,
            "caption_generado": caption,
            "prediccion": prediccion,
            "acierto": acierto,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    total = len(rows)
    sensibilidad = n_correct / total if total else 0.0

    print("\n" + "=" * 70)
    print("RESULTADO -- VALIDACION EXTERNA (dataset nunca visto en entrenamiento)")
    print("=" * 70)
    print(f"Total de imagenes evaluadas: {total}")
    print(f"Detectadas correctamente como 'leucemica': {n_correct}")
    print(f"Sensibilidad externa: {sensibilidad:.4f} ({100 * sensibilidad:.1f}%)")
    print(f"\nCompara este numero contra tu accuracy interno (95%+ de train_binary_classifier.py, "
          f"o el nivel de acierto que veas en tu propio dataset de prueba) -- una caida grande "
          f"aqui confirma el mismo patron de sesgo de dominio que el paper original reporta "
          f"(85% interno -> 52% externo).")
    print(f"\nDetalle completo guardado en: {args.out_csv}")


if __name__ == "__main__":
    main()
