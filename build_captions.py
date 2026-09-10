r"""
build_captions.py

Construye healthy_captions.csv y leukemic_captions.csv (formato image_path,caption)
a partir de:
  - los CSV de atributos oficiales de WBCAtt (pbc_attr_v1_*.csv) para las celulas SANAS
  - los labels/atributos oficiales de LeukemiaAttri para las celulas LEUCEMICAS

IMPORTANTE: los nombres de columnas de abajo (WBCATT_COLS / LEUK_COLS) son los que
documenta cada dataset en su repo/paper. Antes de correr esto, abre tus CSV
descargados y confirma que los nombres de columna coinciden -- si no, ajusta los
diccionarios COLS de mas abajo (es la unica parte que deberias tocar).

Uso (PowerShell):
    python build_captions.py `
        --wbcatt_csv .\attrs\pbc_attr_v1_train.csv `
        --wbcatt_images_dir .\data\healthy `
        --leuk_csv .\attrs\leukemiaattri_labels.csv `
        --leuk_images_dir .\data\leukemic `
        --out_dir .\data
"""

import argparse
import os
import pandas as pd


# --------------------------------------------------------------------------- #
# Mapeo de columnas -- AJUSTA ESTO segun los headers reales de tus CSV
# --------------------------------------------------------------------------- #
WBCATT_COLS = {
    "filename": "path",              # columna con el nombre/ruta de la imagen
    "cell_size": "cell_size",
    "cell_shape": "cell_shape",
    "nucleus_shape": "nucleus_shape",
    "nc_ratio": "nuclear_cytoplasmic_ratio",
    "chromatin": "chromatin_density",
    "vacuole": "cytoplasm_vacuole",
    "cyto_texture": "cytoplasm_texture",
    "cyto_color": "cytoplasm_colour",
    "granule_type": "granule_type",
    "granule_color": "granule_colour",
    "granularity": "granularity",
    "cell_type": "label",            # tipo de celula (neutrofilo, linfocito, etc.)
}

LEUK_COLS = {
    "filename": "filename",
    "cell_size": "cell_size",
    "chromatin": "nuclear_chromatin",
    "nucleus_shape": "nuclear_shape",
    "nucleolus": "nucleolus",
    "cytoplasm": "cytoplasm",
    "basophilia": "cytoplasmic_basophilia",
    "vacuole": "cytoplasmic_vacuoles",
    "diagnosis": "diagnosis",        # ALL, AML, CLL, CML, APML, Normal...
}


def build_wbcatt_caption(row, cols):
    """Genera una oracion morfologica a partir de los atributos WBCAtt de una fila."""
    def g(key):
        col = cols.get(key)
        return row[col] if col and col in row and pd.notna(row[col]) else None

    parts = []
    cell_type = g("cell_type")
    if cell_type:
        parts.append(f"This is a {str(cell_type).lower()}.")

    descriptors = []
    if g("cell_size"):
        descriptors.append(f"a {str(g('cell_size')).lower()} cell size")
    if g("cell_shape"):
        descriptors.append(f"{str(g('cell_shape')).lower()} overall shape")
    if g("nucleus_shape"):
        descriptors.append(f"a {str(g('nucleus_shape')).lower().replace('_', ' ')} nucleus")
    if g("nc_ratio"):
        descriptors.append(f"{str(g('nc_ratio')).lower()} nuclear-to-cytoplasmic ratio")
    if g("chromatin"):
        descriptors.append(f"{str(g('chromatin')).lower()} packed chromatin")
    if g("cyto_texture"):
        descriptors.append(f"{str(g('cyto_texture')).lower()} cytoplasmic texture")
    if g("cyto_color"):
        descriptors.append(f"{str(g('cyto_color')).lower()} cytoplasm")
    if g("granularity") and str(g("granularity")).lower() in ("yes", "true", "1"):
        gtype = g("granule_type")
        gcolor = g("granule_color")
        gran_desc = "granules"
        if gtype:
            gran_desc = f"{str(gtype).lower()} {gran_desc}"
        if gcolor:
            gran_desc = f"{gran_desc} that stain {str(gcolor).lower()}"
        descriptors.append(gran_desc)
    if g("vacuole") and str(g("vacuole")).lower() in ("yes", "true", "1"):
        descriptors.append("visible cytoplasmic vacuoles")

    if descriptors:
        parts.append("It shows " + ", ".join(descriptors) + ".")

    return " ".join(parts) if parts else "A healthy white blood cell with unremarkable morphology."


def build_leuk_caption(row, cols):
    """Genera una oracion morfologica a partir de los atributos LeukemiaAttri de una fila."""
    def g(key):
        col = cols.get(key)
        return row[col] if col and col in row and pd.notna(row[col]) else None

    parts = []
    diagnosis = g("diagnosis")
    if diagnosis:
        parts.append(f"This cell shows features consistent with {diagnosis}.")

    descriptors = []
    if g("cell_size"):
        descriptors.append(f"{str(g('cell_size')).lower()} cell size")
    if g("chromatin"):
        descriptors.append(f"{str(g('chromatin')).lower()} nuclear chromatin")
    if g("nucleus_shape"):
        descriptors.append(f"a {str(g('nucleus_shape')).lower().replace('_', ' ')} nuclear shape")
    if g("nucleolus"):
        descriptors.append(f"{str(g('nucleolus')).lower()} nucleoli")
    if g("cytoplasm"):
        descriptors.append(f"{str(g('cytoplasm')).lower()} cytoplasm")
    if g("basophilia"):
        descriptors.append(f"{str(g('basophilia')).lower()} cytoplasmic basophilia")
    if g("vacuole") and str(g("vacuole")).lower() in ("yes", "true", "1"):
        descriptors.append("cytoplasmic vacuoles")

    if descriptors:
        parts.append("Morphological features include " + ", ".join(descriptors) + ".")

    return " ".join(parts) if parts else "A leukemic white blood cell with abnormal morphology."


def process(csv_path, images_dir, cols, caption_fn, out_csv, strip_first_segment=False):
    df = pd.read_csv(csv_path)
    filename_col = cols["filename"]

    rows = []
    missing = 0
    for _, row in df.iterrows():
        raw_path = str(row[filename_col]).replace("\\", "/")
        if strip_first_segment and "/" in raw_path:
            # Ej: "PBC_dataset_normal_DIB/neutrophil/BNE_7323.jpg" -> "neutrophil/BNE_7323.jpg"
            rel_path = raw_path.split("/", 1)[1]
        else:
            rel_path = os.path.basename(raw_path)
        img_path = os.path.join(images_dir, rel_path)
        if not os.path.exists(img_path):
            missing += 1
            continue
        caption = caption_fn(row, cols)
        rows.append({"image_path": img_path, "caption": caption})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(out_df)} pares imagen-caption escritos. "
          f"{missing} filas del CSV no encontraron imagen correspondiente en {images_dir}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbcatt_csv", type=str, default=None)
    parser.add_argument("--wbcatt_images_dir", type=str, default=None)
    parser.add_argument("--leuk_csv", type=str, default=None)
    parser.add_argument("--leuk_images_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.wbcatt_csv and args.wbcatt_images_dir:
        process(
            args.wbcatt_csv, args.wbcatt_images_dir, WBCATT_COLS,
            build_wbcatt_caption, os.path.join(args.out_dir, "healthy_captions.csv"),
            strip_first_segment=True,  # la columna "path" trae "PBC_dataset_normal_DIB/<tipo>/<archivo>"
        )

    if args.leuk_csv and args.leuk_images_dir:
        process(
            args.leuk_csv, args.leuk_images_dir, LEUK_COLS,
            build_leuk_caption, os.path.join(args.out_dir, "leukemic_captions.csv"),
        )

    if not (args.wbcatt_csv or args.leuk_csv):
        print("No se paso --wbcatt_csv ni --leuk_csv. No hay nada que hacer.")


if __name__ == "__main__":
    main()
