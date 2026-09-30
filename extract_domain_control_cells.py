r"""
extract_domain_control_cells.py

Extrae el grupo de CONTROL DE DOMINIO: celulas de tipo normal (neutrophil,
lymphocyte, monocyte, eosinophil, basophil) que aparecen incidentalmente
dentro de las imagenes de LeukemiaAttri diagnosticadas como AML/ALL.

Estas celulas comparten el mismo microscopio, tincion y protocolo que los
blastos leucemicos (mismo dominio de imagen), pero son morfologicamente
NORMALES. Sirven para probar si un clasificador sana/leucemica aprendio
morfologia real o solo diferencias de dominio/adquisicion de imagen entre
WBCAtt y LeukemiaAttri.

Uso (PowerShell):
    python extract_domain_control_cells.py `
        --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
        --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train `
        --crops_dir .\data\domain_control_crops `
        --out_csv .\data\domain_control_train.csv

    (repite con test.json / Images\test / domain_control_test.csv, y junta
    los CSV igual que has hecho con los demas)
"""

import argparse
import json
import os
import re
from PIL import Image


ALLOWED_DIAGNOSES = {"AML", "ALL"}
NORMAL_CATEGORIES = {"neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"}

DIAGNOSIS_PATTERN = re.compile(r"_(ALL|AML|CLL|CML|APML|Normal)\.", re.IGNORECASE)


def extract_diagnosis(file_name):
    match = DIAGNOSIS_PATTERN.search(file_name)
    return match.group(1).upper() if match else None


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
        if category_name not in NORMAL_CATEGORIES:
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

        rows.append((crop_path, category_name, diagnosis))

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,cell_type,source_diagnosis\n")
        for path, cell_type, diagnosis in rows:
            f.write(f'"{path}","{cell_type}","{diagnosis}"\n')

    print(f"{args.out_csv}: {len(rows)} celulas de control de dominio extraidas "
          f"(normales, dentro de imagenes AML/ALL).")
    print(f"Omitidas por diagnostico fuera de alcance: {skipped_diagnosis}")
    print(f"Omitidas por tipo de celula (blastos/artefactos/ambiguas, no control): {skipped_category}")
    print(f"Omitidas por imagen no encontrada/corrupta: {skipped_no_image}")


if __name__ == "__main__":
    main()