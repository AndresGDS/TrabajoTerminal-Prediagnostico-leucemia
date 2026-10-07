r"""
analyze_detection_metrics.py

Mide que tan bien el sistema detecta blastos usando captions que YA se
generaron (no corre ningun modelo, tarda segundos). Aplica la misma regla
que api_llm.py (palabras clave de tipo de celula blasto sobre el caption en
ingles) y la compara contra el tipo de celula real anotado (columna
cell_type), que es la verdad de terreno por celula.

Reporta sensibilidad, especificidad, precision y F1 del "es un blasto",
y la tasa de alarma por cada tipo de celula real, para ver DONDE falla.

Uso con evaluation_results*.csv (ya trae cell_type y caption generado):
    python analyze_detection_metrics.py `
        --results_csv .\evaluation_results_lora_only.csv

Uso con el detalle de compare_encoders.py (no trae cell_type; se une con el
CSV de prueba por image_path y se elige la columna de caption del modelo):
    python analyze_detection_metrics.py `
        --results_csv .\comparacion_encoders_captions_detalle.csv `
        --caption_col caption_fastvit `
        --truth_csv .\data\leukemic_captions_faithful_test.csv
"""

import argparse
import re

import pandas as pd


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast",
                  "abnormal promyelocyte", "promyelocyte"]


def has_blast_word(text):
    lower = str(text).lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in BLAST_KEYWORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--caption_col", default="caption",
                        help="Columna con el caption GENERADO por el modelo")
    parser.add_argument("--truth_csv", default=None,
                        help="CSV de prueba con image_path y cell_type, si results_csv no trae cell_type")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    if "cell_type" not in df.columns:
        if not args.truth_csv:
            raise SystemExit("results_csv no trae 'cell_type'. Pasa --truth_csv con el CSV de prueba.")
        truth = pd.read_csv(args.truth_csv)[["image_path", "cell_type"]]
        df = df.merge(truth, on="image_path", how="inner")

    print(f"Celulas evaluadas: {len(df)}")
    print("\nTipos de celula reales en el conjunto (verifica que la regla de 'blasto' los clasifica bien):")
    print(df["cell_type"].value_counts().to_string())

    df["truth_blast"] = df["cell_type"].astype(str).apply(has_blast_word)
    df["pred_blast"] = df[args.caption_col].apply(has_blast_word)

    tp = int((df.truth_blast & df.pred_blast).sum())
    fn = int((df.truth_blast & ~df.pred_blast).sum())
    fp = int((~df.truth_blast & df.pred_blast).sum())
    tn = int((~df.truth_blast & ~df.pred_blast).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    sens = ratio(tp, tp + fn)
    spec = ratio(tn, tn + fp)
    prec = ratio(tp, tp + fp)
    f1 = ratio(2 * prec * sens, prec + sens) if (prec == prec and sens == sens) else float("nan")

    print("\n" + "=" * 62)
    print("DETECCION DE BLASTOS (regla de palabras clave sobre el caption)")
    print("=" * 62)
    print(f"Verdaderos blastos: {tp + fn} | No blastos: {tn + fp}")
    print(f"Sensibilidad (blastos detectados):        {sens:.3f}  ({tp}/{tp + fn})")
    print(f"Especificidad (no blastos bien descartados): {spec:.3f}  ({tn}/{tn + fp})")
    print(f"Precision (alarmas que eran blastos):     {prec:.3f}  ({tp}/{tp + fp})")
    print(f"F1:                                       {f1:.3f}")
    print(f"Exactitud balanceada:                     {(sens + spec) / 2:.3f}")

    print("\nTasa de alarma 'blasto' por tipo de celula REAL:")
    por_tipo = (df.groupby("cell_type")["pred_blast"]
                  .agg(alarmas="sum", total="count"))
    por_tipo["tasa_alarma"] = (por_tipo["alarmas"] / por_tipo["total"]).round(3)
    print(por_tipo.sort_values("tasa_alarma", ascending=False).to_string())

    print("\nNota: los 'no blastos' de este conjunto son celulas normales dentro de "
          "imagenes leucemicas (el caso dificil). No incluye celulas sanas de WBCAtt.")


if __name__ == "__main__":
    main()
