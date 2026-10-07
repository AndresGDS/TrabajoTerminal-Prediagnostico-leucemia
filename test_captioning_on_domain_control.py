r"""
test_captioning_on_domain_control.py

A diferencia de validate_domain_shortcut.py (que prueba un CLASIFICADOR
aparte sobre embeddings congelados), esto prueba el MODELO DE CAPTIONING
real -- el mismo que usa api_llm.py para el diagnostico -- contra el
conjunto de control de dominio. El decoder SI se reentreno con el tercer
dominio mezclado (a diferencia del encoder, que nunca se toco), asi que es
una pregunta distinta y no probada todavia: ¿el texto generado mejora,
aunque el encoder congelado siga sin cambiar?

Uso (PowerShell):
    python test_captioning_on_domain_control.py `
        --model_dir .\checkpoints\hemblip_augmented_domain\hemblip_exact_replica_final `
        --domain_control_csv .\data\domain_control_test.csv `
        --out_csv .\captioning_domain_control_results.csv
"""

import argparse
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast", "abnormal promyelocyte", "promyelocyte"]


def detect_diagnosis_from_celltype(caption_en: str) -> str:
    lower = caption_en.lower()
    for keyword in BLAST_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return "leucemica"
    return "sana"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--domain_control_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="./captioning_domain_control_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.domain_control_csv)
    print(f"Celulas de control a evaluar: {len(df)}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando captions"):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=inputs["pixel_values"], max_length=64, num_beams=4,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        prediccion = detect_diagnosis_from_celltype(caption)

        rows.append({
            "image_path": row["image_path"],
            "cell_type_real": row.get("cell_type", "?"),
            "caption_generado": caption,
            "prediccion": prediccion,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    n_total = len(out_df)
    n_correct = (out_df["prediccion"] == "sana").sum()  # "sana" es lo correcto aqui

    print("\n" + "=" * 70)
    print("RESULTADO -- MODELO DE CAPTIONING REAL sobre control de dominio")
    print("=" * 70)
    print(f"Total evaluadas: {n_total}")
    print(f"Correctamente descritas como 'sana' (texto generado sin mencion de blasto): "
          f"{n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"Incorrectamente mencionando un tipo de blasto: {n_total - n_correct} "
          f"({100*(n_total-n_correct)/n_total:.1f}%)")
    print(f"\nCompara esto contra el 0% de 'sana' que dio el clasificador aparte "
          f"(validate_domain_shortcut.py) -- si aqui el numero es mayor a 0%, "
          f"significa que el decoder SI aprendio algo del tercer dominio, aunque "
          f"el encoder congelado siga sin cambiar.")
    print(f"\nDetalle guardado en: {args.out_csv}")


if __name__ == "__main__":
    main()
