r"""
verify_leukemic_images.py

Verifica que todas las imagenes referenciadas en el JSON COCO de
LeukemiaAttri (train.json / test.json) existan realmente en disco.
Util para confirmar que una descarga/extraccion quedo completa antes de
reconstruir los captions.

Uso (PowerShell):
    python verify_leukemic_images.py `
        --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
        --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    total = len(images)
    missing = []

    for img in images:
        path = os.path.join(args.images_dir, img["file_name"])
        if not os.path.exists(path):
            missing.append(img["file_name"])

    print(f"JSON: {args.json}")
    print(f"Carpeta de imagenes: {args.images_dir}")
    print(f"Total de imagenes esperadas segun el JSON: {total}")
    print(f"Imagenes encontradas en disco: {total - len(missing)}")
    print(f"Imagenes FALTANTES: {len(missing)}")

    if missing:
        print("\nPrimeras 10 imagenes faltantes (para diagnosticar el patron):")
        for name in missing[:10]:
            print(f"  - {name}")
        print("\nADVERTENCIA: la extraccion/descarga parece incompleta todavia.")
    else:
        print("\nTodo OK -- ninguna imagen falta. Puedes reconstruir los captions con confianza.")


if __name__ == "__main__":
    main()
