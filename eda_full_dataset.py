r"""
eda_full_dataset.py

EDA exhaustivo, celula por celula, sobre TODO el dataset (no una muestra),
pensado para el protocolo del TT donde el framing es binario: sana vs.
leucemica. Usa los CSV finales (healthy_captions.csv / leukemic_captions.csv)
que ya apuntan a imagenes reales por celula.

Para cada imagen revisa:
  - Si abre correctamente (deteccion de corrupcion)
  - Modo de color (RGB, L=escala de grises, RGBA, CMYK, etc.)
  - Resolucion (ancho x alto)
  - Tamano de archivo en disco (KB)
  - Hash MD5 del contenido, para detectar duplicados exactos
  - Metadatos extra inusuales (perfil ICC, EXIF, transparencia)

Genera:
  - Un CSV detallado (una fila por imagen) para inspeccion manual
  - Graficas de distribucion (modo de color, resolucion, tamano de archivo)
  - Resumen impreso en consola, incluyendo el balance binario sana/leucemica

Uso (PowerShell):
    python eda_full_dataset.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --out_dir .\eda_full_report

Nota: revisa cada imagen del dataset (miles de archivos), puede tardar unos
minutos. No requiere GPU.
"""

import argparse
import hashlib
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


REPORT_LINES = []


def log(msg=""):
    """Imprime en consola Y guarda la linea para el reporte de texto final."""
    print(msg)
    REPORT_LINES.append(str(msg))


def md5_of_file(path, block_size=65536):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            md5.update(block)
    return md5.hexdigest()


def inspect_image(path):
    """Devuelve un dict con toda la info relevante de una imagen, o marca corrupt=True."""
    result = {
        "path": path,
        "exists": os.path.exists(path),
        "corrupt": False,
        "mode": None,
        "width": None,
        "height": None,
        "file_size_kb": None,
        "md5": None,
        "has_icc_profile": False,
        "has_exif": False,
        "has_transparency": False,
    }
    if not result["exists"]:
        result["corrupt"] = True
        return result

    result["file_size_kb"] = round(os.path.getsize(path) / 1024, 2)

    try:
        with Image.open(path) as img:
            img.verify()  # chequeo rapido de integridad
        with Image.open(path) as img:  # reabrir, verify() deja el objeto inutilizable
            result["mode"] = img.mode
            result["width"], result["height"] = img.size
            result["has_icc_profile"] = "icc_profile" in img.info
            result["has_exif"] = "exif" in img.info
            result["has_transparency"] = "transparency" in img.info or img.mode in ("RGBA", "LA")
        result["md5"] = md5_of_file(path)
    except Exception:
        result["corrupt"] = True

    return result


def analyze_group(csv_path, group_name, out_dir):
    log("\n" + "=" * 70)
    log(f"EDA COMPLETO -- {group_name}")
    log("=" * 70)

    df = pd.read_csv(csv_path)
    log(f"Total de filas en {os.path.basename(csv_path)}: {len(df)}")

    records = []
    for path in tqdm(df["image_path"], desc=f"Inspeccionando {group_name}"):
        records.append(inspect_image(path))

    result_df = pd.DataFrame(records)
    detail_csv = os.path.join(out_dir, f"{group_name}_image_details.csv")
    result_df.to_csv(detail_csv, index=False)
    log(f"Detalle guardado en: {detail_csv}")

    n_corrupt = result_df["corrupt"].sum()
    n_missing = (~result_df["exists"]).sum()
    log(f"\nImagenes corruptas o no abribles: {n_corrupt}")
    log(f"Imagenes faltantes (path no existe): {n_missing}")

    valid_df = result_df[~result_df["corrupt"]]

    if len(valid_df) > 0:
        log(f"\nDistribucion de modo de color ({len(valid_df)} imagenes validas):")
        mode_counts = valid_df["mode"].value_counts()
        log(mode_counts)
        if not (mode_counts.index == "RGB").all():
            n_non_rgb = len(valid_df) - mode_counts.get("RGB", 0)
            log(f"ADVERTENCIA: {n_non_rgb} imagenes NO estan en RGB puro "
                  f"(revisar antes de entrenar, BlipProcessor las convierte a RGB "
                  f"automaticamente pero puede perder informacion si eran escala de grises).")

        log(f"\nResolucion -- valores unicos: {valid_df[['width','height']].drop_duplicates().shape[0]}")
        res_counts = valid_df.groupby(["width", "height"]).size().sort_values(ascending=False)
        log(res_counts.head(10))

        log(f"\nTamano de archivo (KB) -- min: {valid_df['file_size_kb'].min():.1f}, "
              f"max: {valid_df['file_size_kb'].max():.1f}, "
              f"promedio: {valid_df['file_size_kb'].mean():.1f}")

        n_icc = valid_df["has_icc_profile"].sum()
        n_exif = valid_df["has_exif"].sum()
        n_transparency = valid_df["has_transparency"].sum()
        log(f"\nMetadatos inusuales -- con perfil ICC: {n_icc}, con EXIF: {n_exif}, "
              f"con transparencia: {n_transparency}")

        n_duplicates = valid_df["md5"].duplicated().sum()
        log(f"\nImagenes duplicadas (mismo contenido exacto, por hash MD5): {n_duplicates}")
        if n_duplicates > 0:
            dup_hashes = valid_df[valid_df["md5"].duplicated(keep=False)].sort_values("md5")
            dup_csv = os.path.join(out_dir, f"{group_name}_duplicates.csv")
            dup_hashes.to_csv(dup_csv, index=False)
            log(f"Detalle de duplicados guardado en: {dup_csv}")

        # Graficas
        plt.figure(figsize=(5, 4))
        mode_counts.plot(kind="bar")
        plt.title(f"Modo de color -- {group_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{group_name}_color_mode.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        valid_df["file_size_kb"].hist(bins=40)
        plt.title(f"Distribucion de tamano de archivo (KB) -- {group_name}")
        plt.xlabel("KB")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{group_name}_file_size_hist.png"))
        plt.close()

    return len(df), int(n_corrupt), int(n_missing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./eda_full_report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_healthy, corrupt_h, missing_h = analyze_group(args.healthy_csv, "sanas", args.out_dir)
    n_leukemic, corrupt_l, missing_l = analyze_group(args.leukemic_csv, "leucemicas", args.out_dir)

    log("\n" + "=" * 70)
    log("RESUMEN GLOBAL -- FRAMING BINARIO (sana vs leucemica)")
    log("=" * 70)
    total = n_healthy + n_leukemic
    log(f"Sanas:      {n_healthy} ({100 * n_healthy / total:.1f}%) -- {corrupt_h} corruptas/faltantes")
    log(f"Leucemicas: {n_leukemic} ({100 * n_leukemic / total:.1f}%) -- {corrupt_l} corruptas/faltantes")
    log(f"Total utilizable: {total - corrupt_h - corrupt_l} de {total}")

    plt.figure(figsize=(4, 4))
    plt.pie([n_healthy - corrupt_h, n_leukemic - corrupt_l],
            labels=["Sanas", "Leucemicas"], autopct="%1.1f%%")
    plt.title("Balance binario (solo imagenes utilizables)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "balance_binario_final.png"))
    plt.close()

    log(f"\nTodo el reporte (CSV + graficas) guardado en: {args.out_dir}")

    summary_path = os.path.join(args.out_dir, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    print(f"\nResumen de texto guardado en: {summary_path}")


if __name__ == "__main__":
    main()
