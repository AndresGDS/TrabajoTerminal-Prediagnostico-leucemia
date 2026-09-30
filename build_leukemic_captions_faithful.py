r"""
build_leukemic_captions_faithful.py

Version FIEL al repositorio real del paper: incluye las 5 subtipos de
leucemia (AML, ALL, CML, CLL, APML) y TODOS los tipos de celula reales
(no solo blastos) -- exactamente el alcance que usa eindresultaat_dataset.csv
en el repositorio original (7,037 sanas + 7,622 leucemicas = 14,659 filas).

A diferencia de build_leukemic_captions_coco.py (nuestra version filtrada,
mas defendible metodologicamente pero infiel al paper), este script
reproduce el comportamiento real: TODA celula anotada dentro de una imagen
diagnosticada hereda ese diagnostico como label -- incluyendo celulas
morfologicamente normales que aparecen incidentalmente en el frotis. Esto
es identico a como lo hace el repositorio real (confirmado inspeccionando
su CSV final), pero SI carga el mismo riesgo de sesgo de dominio que ya
detectaste con validate_domain_shortcut.py -- se documenta como limitacion
heredada del metodo original, no como error propio.

El resultado sigue sirviendo directo para el framing binario (sana vs
leucemica) que es tu objetivo final: cualquier fila de este CSV es
"leucemica" independientemente del subtipo, tal como ya la trata
train_binary_classifier.py.

Uso (PowerShell):
    python build_leukemic_captions_faithful.py `
        --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
        --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train `
        --crops_dir .\data\leukemic_crops_faithful `
        --out_csv .\data\leukemic_captions_faithful_train.csv
"""

import argparse
import json
import os
import re
from PIL import Image


# Alcance FIEL al repositorio real: las 5 subtipos de leucemia, todos los
# tipos de celula reales (se excluye solo "none", que son artefactos/no-celulas).
ALLOWED_DIAGNOSES = {"AML", "ALL", "CML", "CLL", "APML"}
EXCLUDED_CATEGORIES = {"none"}  # unico tipo excluido: no son celulas reales

DIAGNOSIS_PATTERN = re.compile(r"_(ALL|AML|CLL|CML|APML|Normal)\.", re.IGNORECASE)

DIAGNOSIS_FULL_NAME = {
    "AML": "acute myeloid leukemia",
    "ALL": "acute lymphoblastic leukemia",
    "CML": "chronic myeloid leukemia",
    "CLL": "chronic lymphocytic leukemia",
    "APML": "acute promyelocytic leukemia",
}

# Mismo vocabulario que en build_leukemic_captions_coco.py -- pendiente de
# confirmar al 100%, ver esa nota ahi.
ATTR_VALUE_LABELS = {
    "cell_size": {0: "small", 1: "medium", 2: "large"},
    "nuclear_chromatio": {0: "open", 1: "dense"},
    "nuclear_shape": {0: "regular", 1: "irregular"},
    "nucleolus": {0: "absent", 1: "present", 2: "prominent"},
    "cytoplasm": {0: "scanty", 1: "moderate", 2: "abundant"},
    "cytoplasmic_basophilia": {0: "absent", 1: "slight", 2: "moderate", 3: "marked"},
    "cytoplasmic_vacuoles": {0: "absent", 1: "present"},
}

ATTR_DISPLAY_NAME = {
    "cell_size": "size",
    "nuclear_chromatio": "nuclear chromatin",
    "nuclear_shape": "nuclear shape",
    "nucleolus": "nucleolus",
    "cytoplasm": "cytoplasm",
    "cytoplasmic_basophilia": "basophilia",
    "cytoplasmic_vacuoles": "cytoplasmic vacuoles",
}


def extract_diagnosis(file_name):
    match = DIAGNOSIS_PATTERN.search(file_name)
    return match.group(1).upper() if match else None


def attr_text(attr_key, value):
    labels = ATTR_VALUE_LABELS.get(attr_key, {})
    return labels.get(value, f"level {value}")


def build_description(ann, category_name):
    """Formato calcado del repositorio real: '{Tipo} cell. A cell with {attrs}.'"""
    descriptors = []
    for attr_key in ATTR_DISPLAY_NAME:
        if attr_key in ann and ann[attr_key] is not None:
            text = attr_text(attr_key, ann[attr_key])
            display = ATTR_DISPLAY_NAME[attr_key]
            descriptors.append(f"{text} {display}")
    body = ", ".join(descriptors)
    return f"{category_name.capitalize()} cell. A cell with {body}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--crops_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.crops_dir, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    rows = []
    skipped_diagnosis = 0
    skipped_category = 0
    skipped_no_image = 0

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
        if category_name in EXCLUDED_CATEGORIES or not category_name:
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

        description = build_description(ann, category_name)
        diagnosis_full = DIAGNOSIS_FULL_NAME[diagnosis]
        caption = f"{diagnosis_full} {category_name}"  # estilo corto, como el repo real

        rows.append({
            "image_path": crop_path,
            "caption": description,          # caption "rico" en atributos, usado para entrenar
            "diagnosis": diagnosis,           # subtipo exacto (AML/ALL/CML/CLL/APML)
            "cell_type": category_name,
            "label": "leukemia",              # binario: toda fila de este CSV es "leucemica"
            "short_caption": caption,         # estilo corto tipo repo real, por si se necesita
        })

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,caption,diagnosis,cell_type,label,short_caption\n")
        for r in rows:
            def esc(s):
                return str(s).replace('"', '""')
            f.write(
                f'"{esc(r["image_path"])}","{esc(r["caption"])}",'
                f'"{esc(r["diagnosis"])}","{esc(r["cell_type"])}",'
                f'"{esc(r["label"])}","{esc(r["short_caption"])}"\n'
            )

    print(f"{args.out_csv}: {len(rows)} celulas extraidas (alcance FIEL: 5 subtipos, todos los tipos de celula).")
    print(f"Omitidas por diagnostico (Normal/no reconocido): {skipped_diagnosis}")
    print(f"Omitidas por categoria (solo 'none'/artefactos): {skipped_category}")
    print(f"Omitidas por imagen no encontrada/corrupta: {skipped_no_image}")


if __name__ == "__main__":
    main()
