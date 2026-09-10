r"""
build_leukemic_captions_coco.py

LeukemiaAttri viene en formato COCO: cada imagen (640x640) contiene VARIAS celulas
anotadas (una entrada en "annotations" por celula, con su propio bbox y sus 7
atributos morfologicos numericos). Para construir pares imagen-caption a nivel de
celula (como en HemBLIP), este script:

  1. Lee el JSON COCO (train.json / test.json).
  2. Recorta cada celula de la imagen original usando su "bbox" [x, y, w, h].
  3. Guarda el recorte como una imagen nueva (una por celula).
  4. Genera un caption templado a partir de category_name + los 7 atributos.
  5. Escribe leukemic_captions.csv con columnas image_path,caption.

PENDIENTE: el diccionario ATTR_VALUE_LABELS de abajo traduce cada valor numerico
(0, 1, 2...) a texto. No encontre el legend oficial en el paper principal (esta
en el material suplementario). Por ahora dejo placeholders "nivel 0"/"nivel 1".
Si encuentras el legend real (en la carpeta de Drive o en classify/utils del repo),
solo reemplaza los valores del diccionario -- el resto del script no cambia.

Uso (PowerShell):
    python build_leukemic_captions_coco.py `
        --json .\LeukemiaAttri\Annotations\train.json `
        --images_dir .\LeukemiaAttri\Images\train `
        --crops_dir .\data\leukemic_crops `
        --out_csv .\data\leukemic_captions.csv
"""

import argparse
import json
import os
import re
from PIL import Image


# Alcance del proyecto: solo leucemias agudas (AML, ALL), excluyendo CML/CLL/APML
# y excluyendo celulas normales o artefactos que aparecen incidentalmente en el
# frotis de un paciente leucemico (ver documentacion_sistema_hemvlm.md, seccion 7).
ALLOWED_DIAGNOSES = {"AML", "ALL"}
ALLOWED_CATEGORIES = {"myeloblast", "lymphoblast"}  # blastos que definen AML y ALL respectivamente

DIAGNOSIS_PATTERN = re.compile(r"_(ALL|AML|CLL|CML|APML|Normal)\.", re.IGNORECASE)


def extract_diagnosis(file_name):
    match = DIAGNOSIS_PATTERN.search(file_name)
    return match.group(1).upper() if match else None


# --------------------------------------------------------------------------- #
# PENDIENTE DE CONFIRMAR: significado real de cada valor numerico por atributo.
# Ajusta esto en cuanto tengas el legend oficial del dataset.
# --------------------------------------------------------------------------- #
ATTR_VALUE_LABELS = {
    "cell_size": {0: "small", 1: "large"},
    "nuclear_chromatio": {0: "open", 1: "dense"},          # nombre tal cual viene en el JSON (typo original)
    "nuclear_shape": {0: "regular", 1: "irregular"},
    "nucleolus": {0: "absent", 1: "present"},
    "cytoplasm": {0: "scant", 1: "abundant"},
    "cytoplasmic_basophilia": {0: "low", 1: "high"},
    "cytoplasmic_vacuoles": {0: "absent", 1: "present"},
}

ATTR_DISPLAY_NAME = {
    "cell_size": "cell size",
    "nuclear_chromatio": "nuclear chromatin",
    "nuclear_shape": "nuclear shape",
    "nucleolus": "nucleolus",
    "cytoplasm": "cytoplasm",
    "cytoplasmic_basophilia": "cytoplasmic basophilia",
    "cytoplasmic_vacuoles": "cytoplasmic vacuoles",
}


def attr_text(attr_key, value):
    """Traduce un valor numerico de atributo a texto, o cae a 'level N' si no esta mapeado."""
    labels = ATTR_VALUE_LABELS.get(attr_key, {})
    if value in labels:
        return labels[value]
    return f"level {value}"


DIAGNOSIS_FULL_NAME = {
    "AML": "acute myeloid leukemia (AML)",
    "ALL": "acute lymphoblastic leukemia (ALL)",
}


def build_caption(ann, diagnosis):
    category = ann.get("category_name", "white blood cell")
    diagnosis_text = DIAGNOSIS_FULL_NAME.get(diagnosis, diagnosis)
    parts = [f"This is a {category}, consistent with {diagnosis_text}."]

    descriptors = []
    for attr_key in ATTR_DISPLAY_NAME:
        if attr_key in ann and ann[attr_key] is not None:
            value = ann[attr_key]
            text = attr_text(attr_key, value)
            display = ATTR_DISPLAY_NAME[attr_key]
            descriptors.append(f"{text} {display}")

    if descriptors:
        parts.append("Morphological features: " + ", ".join(descriptors) + ".")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Ruta al train.json o test.json COCO")
    parser.add_argument("--images_dir", type=str, required=True, help="Carpeta con las imagenes completas (640x640)")
    parser.add_argument("--crops_dir", type=str, required=True, help="Carpeta de salida para los recortes por celula")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--padding", type=int, default=0, help="Pixeles extra de margen alrededor del bbox")
    args = parser.parse_args()

    os.makedirs(args.crops_dir, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    rows = []
    skipped_no_image = 0
    skipped_diagnosis = 0
    skipped_category = 0
    for ann in coco["annotations"]:
        image_info = images_by_id.get(ann["image_id"])
        if image_info is None:
            skipped_no_image += 1
            continue

        diagnosis = extract_diagnosis(image_info["file_name"])
        if diagnosis not in ALLOWED_DIAGNOSES:
            skipped_diagnosis += 1
            continue

        category_name = ann.get("category_name", "")
        if category_name not in ALLOWED_CATEGORIES:
            skipped_category += 1
            continue

        src_path = os.path.join(args.images_dir, image_info["file_name"])
        if not os.path.exists(src_path):
            skipped_no_image += 1
            continue

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception:
            skipped_no_image += 1
            continue

        x, y, w, h = ann["bbox"]
        pad = args.padding
        left = max(0, int(x - pad))
        top = max(0, int(y - pad))
        right = min(img.width, int(x + w + pad))
        bottom = min(img.height, int(y + h + pad))
        crop = img.crop((left, top, right, bottom))

        base_name = os.path.splitext(image_info["file_name"])[0]
        crop_name = f"{base_name}_cell{ann.get('cell_id', ann['id'])}.png"
        crop_path = os.path.join(args.crops_dir, crop_name)
        crop.save(crop_path)

        caption = build_caption(ann, diagnosis)
        rows.append((crop_path, caption))

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,caption\n")
        for path, caption in rows:
            # Escapar comillas dobles dentro del caption para no romper el CSV
            safe_caption = caption.replace('"', '""')
            f.write(f'"{path}","{safe_caption}"\n')

    print(f"{args.out_csv}: {len(rows)} recortes de celula generados (alcance: AML/ALL, "
          f"solo myeloblast/lymphoblast).")
    print(f"Omitidas por diagnostico fuera de alcance (CML/CLL/APML/otro): {skipped_diagnosis}")
    print(f"Omitidas por tipo de celula fuera de alcance (none, normales, ambiguas): {skipped_category}")
    print(f"Omitidas por imagen no encontrada/corrupta: {skipped_no_image}")


if __name__ == "__main__":
    main()
