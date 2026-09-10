r"""
eda_dataset.py

Analisis exploratorio del dataset combinado (celulas sanas WBCAtt + celulas
leucemicas LeukemiaAttri) usado para HemVLM. Genera:

  - Distribucion de tipos de celula (sanas y leucemicas)
  - Distribucion de cada atributo morfologico categorico
  - Distribucion de diagnosticos (ALL/AML/CLL/CML/APML) en leucemicas, extraida
    del nombre de archivo
  - Balance sano vs leucemico
  - Chequeo de resolucion de imagenes (muestra aleatoria)
  - Todo se imprime en consola y ademas se guardan las graficas como PNG

Uso (PowerShell):
    python eda_dataset.py `
        --wbcatt_csv .\pbc_attr_v1_train.csv `
        --wbcatt_images_dir .\bloodcells_dataset `
        --leuk_json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
        --leuk_images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train `
        --out_dir .\eda_report
"""

import argparse
import json
import os
import random
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no necesita pantalla, solo guarda archivos
import matplotlib.pyplot as plt
from PIL import Image


def plot_bar(series, title, out_path, top_n=20):
    counts = series.value_counts().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(3, len(counts) * 0.35)))
    counts.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Cantidad")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return counts


def eda_wbcatt(csv_path, images_dir, out_dir):
    print("\n" + "=" * 70)
    print("EDA -- CELULAS SANAS (WBCAtt)")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Filas totales: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    print("\nValores nulos por columna:")
    print(df.isna().sum())

    print("\nDistribucion por tipo de celula (label):")
    counts = plot_bar(df["label"], "Distribucion de tipos de celula (sanas)",
                       os.path.join(out_dir, "wbcatt_label_dist.png"))
    print(counts)

    categorical_attrs = [
        "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
        "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
        "cytoplasm_colour", "granule_type", "granule_colour", "granularity",
    ]
    for attr in categorical_attrs:
        if attr in df.columns:
            print(f"\nDistribucion de '{attr}':")
            counts = plot_bar(df[attr], f"Distribucion de {attr} (sanas)",
                               os.path.join(out_dir, f"wbcatt_{attr}_dist.png"))
            print(counts)

    # Chequeo de resolucion en una muestra de imagenes
    check_image_resolutions(df, images_dir, path_col="path", out_dir=out_dir,
                             prefix="wbcatt", strip_first_segment=True)

    return len(df)


DIAGNOSIS_PATTERN = re.compile(r"_(ALL|AML|CLL|CML|APML|Normal)\.", re.IGNORECASE)


def extract_diagnosis(file_name):
    match = DIAGNOSIS_PATTERN.search(file_name)
    return match.group(1).upper() if match else "desconocido"


def eda_leukemic(json_path, images_dir, out_dir):
    print("\n" + "=" * 70)
    print("EDA -- CELULAS LEUCEMICAS (LeukemiaAttri)")
    print("=" * 70)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    print(f"Imagenes (campos completos): {len(images)}")
    print(f"Anotaciones (celulas individuales): {len(annotations)}")

    ann_df = pd.DataFrame(annotations)
    ann_df["category_name"] = ann_df["category_id"].map(categories)

    print("\nDistribucion por tipo de celula (category_name):")
    counts = plot_bar(ann_df["category_name"], "Distribucion de tipos de celula (leucemicas)",
                       os.path.join(out_dir, "leuk_category_dist.png"))
    print(counts)

    categorical_attrs = [
        "cell_size", "nuclear_chromatio", "nuclear_shape", "nucleolus",
        "cytoplasm", "cytoplasmic_basophilia", "cytoplasmic_vacuoles",
    ]
    for attr in categorical_attrs:
        if attr in ann_df.columns:
            print(f"\nDistribucion de '{attr}' (codigos numericos, legend pendiente de confirmar):")
            counts = plot_bar(ann_df[attr].astype(str), f"Distribucion de {attr} (leucemicas)",
                               os.path.join(out_dir, f"leuk_{attr}_dist.png"))
            print(counts)

    # Diagnostico (ALL/AML/CLL/CML/APML) extraido del nombre de archivo
    images_df = pd.DataFrame(images)
    images_df["diagnosis"] = images_df["file_name"].apply(extract_diagnosis)
    print("\nDistribucion de diagnostico (extraido del nombre de archivo):")
    counts = plot_bar(images_df["diagnosis"], "Distribucion de diagnostico (leucemicas)",
                       os.path.join(out_dir, "leuk_diagnosis_dist.png"))
    print(counts)

    # Cuantas celulas anotadas hay en promedio por imagen de campo completo
    cells_per_image = ann_df.groupby("image_id").size()
    print(f"\nCelulas anotadas por imagen -- promedio: {cells_per_image.mean():.2f}, "
          f"minimo: {cells_per_image.min()}, maximo: {cells_per_image.max()}")

    # Chequeo de resolucion en una muestra de imagenes
    check_image_resolutions(images_df, images_dir, path_col="file_name", out_dir=out_dir,
                             prefix="leuk", strip_first_segment=False)

    return len(annotations)


def check_image_resolutions(df, images_dir, path_col, out_dir, prefix, strip_first_segment, sample_size=50):
    sample = df[path_col].dropna().sample(min(sample_size, len(df)), random_state=42)
    resolutions = []
    missing = 0
    for raw_path in sample:
        raw_path = str(raw_path).replace("\\", "/")
        if strip_first_segment and "/" in raw_path:
            rel_path = raw_path.split("/", 1)[1]
        else:
            rel_path = os.path.basename(raw_path)
        full_path = os.path.join(images_dir, rel_path)
        if not os.path.exists(full_path):
            missing += 1
            continue
        try:
            with Image.open(full_path) as img:
                resolutions.append(img.size)
        except Exception:
            missing += 1

    print(f"\nMuestra de resolucion de imagenes ({prefix}, n={len(resolutions)}, "
          f"{missing} no encontradas/corruptas de {len(sample)}):")
    if resolutions:
        unique_res = pd.Series(resolutions).value_counts()
        print(unique_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbcatt_csv", type=str, default=None)
    parser.add_argument("--wbcatt_images_dir", type=str, default=None)
    parser.add_argument("--leuk_json", type=str, default=None)
    parser.add_argument("--leuk_images_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./eda_report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_healthy = 0
    n_leukemic = 0

    if args.wbcatt_csv and args.wbcatt_images_dir:
        n_healthy = eda_wbcatt(args.wbcatt_csv, args.wbcatt_images_dir, args.out_dir)

    if args.leuk_json and args.leuk_images_dir:
        n_leukemic = eda_leukemic(args.leuk_json, args.leuk_images_dir, args.out_dir)

    if n_healthy and n_leukemic:
        print("\n" + "=" * 70)
        print("BALANCE SANAS vs LEUCEMICAS")
        print("=" * 70)
        total = n_healthy + n_leukemic
        print(f"Sanas:      {n_healthy} ({100 * n_healthy / total:.1f}%)")
        print(f"Leucemicas: {n_leukemic} ({100 * n_leukemic / total:.1f}%)")

        plt.figure(figsize=(4, 4))
        plt.pie([n_healthy, n_leukemic], labels=["Sanas", "Leucemicas"], autopct="%1.1f%%")
        plt.title("Balance sanas vs leucemicas")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "balance_sanas_vs_leucemicas.png"))
        plt.close()

    print(f"\nGraficas guardadas en: {args.out_dir}")


if __name__ == "__main__":
    main()
