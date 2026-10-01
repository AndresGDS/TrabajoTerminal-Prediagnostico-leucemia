r"""
build_aml_cytomorphology_captions.py

Construye un CSV de captions a partir del dataset AML-Cytomorphology_LMU
(TCIA, Matek et al.) -- 18,365 imagenes de celula individual, organizadas
en carpetas por codigo de tipo morfologico. A diferencia de WBCAtt/
LeukemiaAttri, este dataset NO trae atributos morfologicos detallados
(cromatina, nucleolo, etc.) -- solo el tipo de celula. Los captions que
genera este script son por lo tanto simples ("This is a myeloblast."),
no descriptivos con atributos.

Valor real de este dataset: contiene celulas de pacientes con AML Y de
pacientes control (sin neoplasia) capturadas con el MISMO microscopio --
al mezclarlo en el entrenamiento, ambas clases (sana/leucemica) quedan con
ejemplos de una tercera fuente de adquisicion, rompiendo la correlacion
trivial "camara = clase" que causaba el sesgo de dominio detectado
(ver validate_domain_shortcut.py).

Asume la estructura de carpetas estandar de esta coleccion:
    <root>/BAS/BAS_0001.tiff
    <root>/EOS/EOS_0001.tiff
    ... etc, una carpeta por codigo de 2-3 letras.

Uso (PowerShell):
    python build_aml_cytomorphology_captions.py `
        --root_dir .\AML-Cytomorphology_LMU `
        --crops_dir .\data\aml_cytomorphology_crops `
        --out_csv .\data\aml_cytomorphology_captions.csv
"""

import argparse
import os
from PIL import Image


# Codigo de carpeta -> (nombre completo en ingles, es_blasto)
CLASS_INFO = {
    "BAS": ("basophil", False),
    "EBO": ("erythroblast", False),
    "EOS": ("eosinophil", False),
    "KSC": ("smudge cell", False),
    "LYA": ("atypical lymphocyte", False),
    "LYT": ("lymphocyte", False),
    "MMZ": ("metamyelocyte", False),
    "MOB": ("monoblast", True),
    "MON": ("monocyte", False),
    "MYB": ("myelocyte", False),
    "MYO": ("myeloblast", True),
    "NGB": ("band neutrophil", False),
    "NGS": ("segmented neutrophil", False),
    "PMB": ("bilobed promyelocyte", True),
    "PMO": ("promyelocyte", True),
    # UNC (no se pudo clasificar) se omite -- ver skipped_unclassified abajo
}

VALID_EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                         help="Carpeta raiz del dataset, con subcarpetas por codigo de clase (BAS, EOS, etc.)")
    parser.add_argument("--crops_dir", type=str, required=True,
                         help="Donde guardar las imagenes convertidas a formato estandar (PNG)")
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.crops_dir, exist_ok=True)

    rows = []
    skipped_unclassified = 0
    skipped_unknown_folder = 0
    skipped_corrupt = 0

    for entry in os.listdir(args.root_dir):
        folder_path = os.path.join(args.root_dir, entry)
        if not os.path.isdir(folder_path):
            continue

        class_code = entry.strip().upper()
        if class_code == "UNC":
            skipped_unclassified += len([
                f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)
            ])
            continue

        if class_code not in CLASS_INFO:
            print(f"Carpeta no reconocida, se omite: {entry}")
            skipped_unknown_folder += 1
            continue

        class_name, is_blast = CLASS_INFO[class_code]

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue

            src_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception:
                skipped_corrupt += 1
                continue

            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(args.crops_dir, f"{class_code}_{out_name}")
            img.save(out_path)

            caption = f"This is a {class_name}."
            label = "leukemia" if is_blast else "healthy"

            rows.append({
                "image_path": out_path,
                "caption": caption,
                "cell_type_code": class_code,
                "cell_type": class_name,
                "label": label,
            })

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,caption,cell_type_code,cell_type,label\n")
        for r in rows:
            def esc(s):
                return str(s).replace('"', '""')
            f.write(
                f'"{esc(r["image_path"])}","{esc(r["caption"])}",'
                f'"{esc(r["cell_type_code"])}","{esc(r["cell_type"])}","{esc(r["label"])}"\n'
            )

    n_leukemia = sum(1 for r in rows if r["label"] == "leukemia")
    n_healthy = sum(1 for r in rows if r["label"] == "healthy")

    print(f"\n{args.out_csv}: {len(rows)} imagenes procesadas.")
    print(f"  Leucemicas (blastos: MYO/MOB/PMO/PMB): {n_leukemia}")
    print(f"  Sanas (resto de tipos): {n_healthy}")
    print(f"Omitidas por no clasificadas (UNC): {skipped_unclassified}")
    print(f"Omitidas por carpeta no reconocida: {skipped_unknown_folder}")
    print(f"Omitidas por imagen corrupta: {skipped_corrupt}")


if __name__ == "__main__":
    main()
