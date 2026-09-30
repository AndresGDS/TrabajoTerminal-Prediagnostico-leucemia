r"""
build_vqa_pairs.py

Genera pares pregunta-respuesta (VQA) a partir de los CSV de captions que
ya tienes (healthy_captions.csv, leukemic_captions.csv), replicando el
enfoque del repositorio original: en vez de solo generar un caption libre
y luego buscarle palabras clave (fragil, como vimos con el bug de
"small"/"overall"), el modelo se entrena para responder preguntas
directas, incluyendo la pregunta de diagnostico:

    "Is this a healthy or diseased cell?" -> "This is a leukemic blast."
                                           -> "This is a healthy white blood cell."

Uso (PowerShell):
    python build_vqa_pairs.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --out_csv .\data\vqa_pairs.csv
"""

import argparse
import pandas as pd


def build_healthy_vqa(caption: str):
    """A partir de un caption de celula sana, genera preguntas y respuestas."""
    pairs = []
    pairs.append((
        "Is this a healthy or diseased cell?",
        "This is a healthy white blood cell.",
    ))
    # Extrae el tipo de celula de la primera oracion: "This is a X."
    if caption.lower().startswith("this is a"):
        cell_type = caption.split(".")[0].replace("This is a", "").strip()
        pairs.append((
            "What type of white blood cell is shown?",
            cell_type.capitalize() + ".",
        ))
    return pairs


def build_leukemic_vqa(caption: str, diagnosis_text: str):
    """A partir de un caption de celula leucemica, genera preguntas y respuestas."""
    pairs = []
    pairs.append((
        "Is this a healthy or diseased cell?",
        "This is a leukemic blast.",
    ))
    pairs.append((
        "What diagnosis is consistent with this cell?",
        diagnosis_text,
    ))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    healthy_df = pd.read_csv(args.healthy_csv)
    leukemic_df = pd.read_csv(args.leukemic_csv)

    rows = []
    for _, row in healthy_df.iterrows():
        for question, answer in build_healthy_vqa(row["caption"]):
            rows.append({"image_path": row["image_path"], "question": question, "answer": answer})

    for _, row in leukemic_df.iterrows():
        caption = row["caption"]
        # el caption ya incluye "consistent with acute myeloid leukemia (AML)" etc.
        diagnosis_text = "This cell shows features of acute leukemia."
        if "acute myeloid leukemia" in caption.lower():
            diagnosis_text = "This cell is consistent with acute myeloid leukemia (AML)."
        elif "acute lymphoblastic leukemia" in caption.lower():
            diagnosis_text = "This cell is consistent with acute lymphoblastic leukemia (ALL)."
        for question, answer in build_leukemic_vqa(caption, diagnosis_text):
            rows.append({"image_path": row["image_path"], "question": question, "answer": answer})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"{args.out_csv}: {len(out_df)} pares pregunta-respuesta generados "
          f"({len(healthy_df)} sanas x2 preguntas + {len(leukemic_df)} leucemicas x2 preguntas).")


if __name__ == "__main__":
    main()
