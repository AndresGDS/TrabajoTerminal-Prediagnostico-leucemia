r"""
compare_domain_control_results.py

Compara dos corridas de test_captioning_on_domain_control*.py imagen por
imagen (mismas 318 celulas de control) con una prueba de McNemar exacta,
que es la prueba correcta para datos PAREADOS (cada imagen la evaluan
ambos modelos). Solo cuentan las imagenes donde los modelos DISCREPAN.

Uso (PowerShell):
    python compare_domain_control_results.py `
        --csv_a .\captioning_domain_control_results_baseline.csv `
        --csv_b .\captioning_domain_control_results_fastvit.csv `
        --name_a ViT-B/16 `
        --name_b FastViT
"""

import argparse
from math import comb

import pandas as pd


def mcnemar_exact(b, c):
    """p-valor bilateral exacto (binomial, p=0.5) sobre las discordancias."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_a", required=True)
    parser.add_argument("--csv_b", required=True)
    parser.add_argument("--name_a", default="Modelo A")
    parser.add_argument("--name_b", default="Modelo B")
    args = parser.parse_args()

    a = pd.read_csv(args.csv_a)[["image_path", "cell_type_real", "prediccion"]]
    b = pd.read_csv(args.csv_b)[["image_path", "prediccion"]]
    df = a.merge(b, on="image_path", suffixes=("_a", "_b"))

    # "sana" es la respuesta correcta en este conjunto de control
    df["ok_a"] = df["prediccion_a"] == "sana"
    df["ok_b"] = df["prediccion_b"] == "sana"

    n = len(df)
    print(f"Imagenes emparejadas: {n}")
    print(f"{args.name_a}: {df['ok_a'].sum()} correctas ({100 * df['ok_a'].mean():.1f}%)")
    print(f"{args.name_b}: {df['ok_b'].sum()} correctas ({100 * df['ok_b'].mean():.1f}%)")

    both_ok = int((df["ok_a"] & df["ok_b"]).sum())
    both_bad = int((~df["ok_a"] & ~df["ok_b"]).sum())
    only_a = int((df["ok_a"] & ~df["ok_b"]).sum())  # A acierta, B falla
    only_b = int((~df["ok_a"] & df["ok_b"]).sum())  # B acierta, A falla

    print("\nTabla de concordancia:")
    print(f"  Ambos aciertan:                  {both_ok}")
    print(f"  Ambos fallan:                    {both_bad}")
    print(f"  Solo {args.name_a} acierta:  {only_a}")
    print(f"  Solo {args.name_b} acierta:  {only_b}")

    p = mcnemar_exact(only_a, only_b)
    print(f"\nPrueba de McNemar exacta (bilateral): p = {p:.4f}")
    if p < 0.05:
        print("Diferencia estadisticamente significativa (p < 0.05).")
    else:
        print("Diferencia NO significativa a nivel 0.05 -- compatible con ruido de muestreo.")

    print("\nDesglose de errores por tipo de celula (fallos de cada modelo):")
    errores = pd.DataFrame({
        args.name_a: df[~df["ok_a"]]["cell_type_real"].value_counts(),
        args.name_b: df[~df["ok_b"]]["cell_type_real"].value_counts(),
    }).fillna(0).astype(int)
    print(errores)

    print("\nNota: es UNA corrida de entrenamiento por modelo (una semilla); "
          "esta prueba mide el ruido de la muestra de 318 imagenes, no la "
          "variabilidad entre entrenamientos.")


if __name__ == "__main__":
    main()
