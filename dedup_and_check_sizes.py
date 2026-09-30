r"""
dedup_and_check_sizes.py

Limpia duplicados exactos (por hash MD5, ya calculado por eda_full_dataset.py
en los archivos *_image_details.csv) de los CSV de captions, y reporta la
distribucion de dimensiones de imagen para detectar recortes sospechosamente
pequenos (posibles detecciones parciales/artefactos en los bordes).

Uso (PowerShell):
    python dedup_and_check_sizes.py `
        --captions_csv .\data\healthy_captions_faithful.csv `
        --details_csv .\eda_full_report_faithful\sanas_image_details.csv `
        --out_csv .\data\healthy_captions_faithful_dedup.csv

    python dedup_and_check_sizes.py `
        --captions_csv .\data\leukemic_captions_faithful.csv `
        --details_csv .\eda_full_report_faithful\leucemicas_image_details.csv `
        --out_csv .\data\leukemic_captions_faithful_dedup.csv `
        --min_dimension 30
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_csv", type=str, required=True)
    parser.add_argument("--details_csv", type=str, required=True,
                         help="El *_image_details.csv que genero eda_full_dataset.py (tiene el hash md5)")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--min_dimension", type=int, default=0,
                         help="Si se da, tambien descarta imagenes con ancho o alto menor a este valor (en pixeles)")
    args = parser.parse_args()

    captions_df = pd.read_csv(args.captions_csv)
    details_df = pd.read_csv(args.details_csv)

    merged = captions_df.merge(
        details_df[["path", "md5", "width", "height", "file_size_kb"]],
        left_on="image_path", right_on="path", how="left",
    )

    print(f"Filas originales: {len(merged)}")

    # --- Reporte de distribucion de tamanos, antes de filtrar ---
    print("\nDistribucion de dimensiones (percentiles):")
    print(merged[["width", "height"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    n_tiny = ((merged["width"] < 30) | (merged["height"] < 30)).sum()
    print(f"\nImagenes con ancho o alto menor a 30px: {n_tiny} ({100 * n_tiny / len(merged):.2f}%)")

    # --- Quitar duplicados por hash MD5 (se queda con la primera aparicion) ---
    before = len(merged)
    merged = merged.drop_duplicates(subset="md5", keep="first")
    n_dup_removed = before - len(merged)
    print(f"\nDuplicados eliminados (por hash MD5): {n_dup_removed}")

    # --- Opcional: quitar recortes muy pequenos ---
    if args.min_dimension > 0:
        before = len(merged)
        merged = merged[(merged["width"] >= args.min_dimension) & (merged["height"] >= args.min_dimension)]
        n_tiny_removed = before - len(merged)
        print(f"Recortes menores a {args.min_dimension}px eliminados: {n_tiny_removed}")

    # Quitar columnas auxiliares antes de guardar
    out_cols = [c for c in captions_df.columns]
    merged[out_cols].to_csv(args.out_csv, index=False)

    print(f"\nFilas finales: {len(merged)}")
    print(f"Guardado en: {args.out_csv}")


if __name__ == "__main__":
    main()
