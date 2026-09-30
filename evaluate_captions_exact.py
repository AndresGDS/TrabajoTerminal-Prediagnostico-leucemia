r"""
evaluate_captions_exact.py

Evaluacion cuantitativa de los captions generados, replicando EXACTAMENTE
la metodologia del repositorio original (evaluate_captions.py):
  - BLEU con NLTK (sentence_bleu + SmoothingFunction method4)
  - ROUGE-1 y ROUGE-L (rouge_score, con stemmer)
  - BERTScore usando BioBERT (dmis-lab/biobert-base-cased-v1.1), no un
    BERT generico -- mas apropiado para texto biomedico.

Requiere: pip install nltk rouge-score bert-score

Uso (PowerShell):
    python evaluate_captions_exact.py `
        --model_dir .\checkpoints\hemblip_base_single_stage_v2\hemblip_base_single_stage_final `
        --test_csv .\data\leukemic_captions_test.csv `
        --out_csv .\evaluation_results.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import BlipProcessor, BlipForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True,
                         help="CSV con columnas image_path y caption (la referencia)")
    parser.add_argument("--out_csv", type=str, default="./evaluation_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    df = pd.read_csv(args.test_csv)
    df = df.rename(columns={"caption": "true_description"})

    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    hypotheses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando captions"):
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=inputs["pixel_values"], max_length=64, num_beams=4,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        hypotheses.append(processor.decode(output_ids[0], skip_special_tokens=True))

    df["caption"] = hypotheses
    references = df["true_description"].astype(str).tolist()

    print(f"Evaluando {len(hypotheses)} predicciones...")

    # === BLEU ===
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
        for ref, hyp in zip(references, hypotheses)
    ]
    print(f"Mean BLEU: {np.mean(bleu_scores):.4f}")

    # === ROUGE ===
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores, rougeL_scores = [], []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
    print(f"Mean ROUGE-1: {np.mean(rouge1_scores):.4f}")
    print(f"Mean ROUGE-L: {np.mean(rougeL_scores):.4f}")

    df["caption"] = hypotheses
    df["bleu"] = bleu_scores
    df["rouge1"] = rouge1_scores
    df["rougeL"] = rougeL_scores
    df.to_csv(args.out_csv, index=False)
    print(f"Resultados parciales (BLEU/ROUGE) guardados en: {args.out_csv}")
    print("(si BERTScore falla ahora, ya no se pierde el trabajo de generacion + BLEU/ROUGE)")

    # === BERTScore con BioBERT ===
    print("Computando BERTScore (BioBERT)...")
    P, R, F1 = bert_score(
        cands=hypotheses, refs=references, lang="en",
        model_type="dmis-lab/biobert-base-cased-v1.1", num_layers=12, idf=False,
        device=device, use_fast_tokenizer=False,
    )
    print(f"Mean BERTScore (F1): {F1.mean():.4f}")

    df["bertscore_f1"] = F1.tolist()
    df.to_csv(args.out_csv, index=False)
    print(f"Resultados completos guardados en: {args.out_csv}")


if __name__ == "__main__":
    main()
