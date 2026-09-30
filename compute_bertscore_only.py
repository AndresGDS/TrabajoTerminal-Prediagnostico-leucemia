r"""
compute_bertscore_only.py

Calcula el BERTScore (BioBERT) sobre un evaluation_results.csv que ya tiene
las columnas 'true_description' (referencia) y 'caption' (generado), sin
necesidad de volver a generar los captions. Usa el tokenizer "lento" de
BioBERT (use_fast_tokenizer=False) para evitar el error de conversion que
pide sentencepiece/tiktoken.

Uso (PowerShell):
    python compute_bertscore_only.py --csv .\evaluation_results.csv
"""

import argparse
import pandas as pd
import torch
from bert_score import score as bert_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    df = pd.read_csv(args.csv)
    references = df["true_description"].astype(str).tolist()
    hypotheses = df["caption"].astype(str).tolist()

    print(f"Calculando BERTScore sobre {len(hypotheses)} pares...")
    P, R, F1 = bert_score(
        cands=hypotheses, refs=references, lang="en",
        model_type="dmis-lab/biobert-base-cased-v1.1", num_layers=12, idf=False,
        device=device, use_fast_tokenizer=False,
    )

    df["bertscore_f1"] = F1.tolist()
    df.to_csv(args.csv, index=False)

    print(f"\nMean BERTScore (F1): {F1.mean():.4f}")
    print(f"Guardado (actualizado) en: {args.csv}")


if __name__ == "__main__":
    main()
