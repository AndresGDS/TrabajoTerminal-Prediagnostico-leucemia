r"""
validate_domain_shortcut.py

Corre el clasificador binario ya entrenado (train_binary_classifier.py)
sobre el grupo de CONTROL DE DOMINIO (celulas normales dentro de imagenes
leucemicas, ver extract_domain_control_cells.py).

Interpretacion del resultado:
  - Si el clasificador dice mayormente "sana" para estas celulas -> buena
    senal, sugiere que aprendio morfologia real (reconoce que una celula
    normal es normal, aunque venga del dominio de imagen "leucemico").
  - Si el clasificador dice mayormente "leucemica" -> mala senal, confirma
    que aprendio a distinguir el DOMINIO de la imagen (de que dataset/
    microscopio viene), no la morfologia celular.

Uso (PowerShell):
    python validate_domain_shortcut.py `
        --model_dir .\checkpoints\hemblip_base_single_stage_v2\hemblip_base_single_stage_final `
        --classifier_path .\checkpoints\binary_classifier.joblib `
        --domain_control_csv .\data\domain_control_train.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


@torch.no_grad()
def extract_embedding(vision_model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = vision_model(pixel_values=inputs["pixel_values"])
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, required=True)
    parser.add_argument("--domain_control_csv", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando encoder visual y clasificador...")
    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    vision_model = model.vision_model
    clf = joblib.load(args.classifier_path)

    df = pd.read_csv(args.domain_control_csv)
    print(f"Celulas de control de dominio a evaluar: {len(df)}")

    embeddings = []
    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo embeddings"):
        try:
            emb = extract_embedding(vision_model, processor, row["image_path"], device)
            embeddings.append(emb)
            valid_rows.append(row)
        except Exception:
            continue

    X = np.array(embeddings)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]  # probabilidad de "leucemica"

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    valid_df["prediccion"] = np.where(preds == 1, "leucemica", "sana")
    valid_df["prob_leucemica"] = probs

    n_total = len(valid_df)
    n_pred_leucemica = (preds == 1).sum()
    n_pred_sana = (preds == 0).sum()

    print("\n" + "=" * 60)
    print("RESULTADO -- CONTROL DE DOMINIO (celulas normales en imagenes AML/ALL)")
    print("=" * 60)
    print(f"Total evaluadas: {n_total}")
    print(f"Predichas como 'sana' (esperado si aprendio morfologia): {n_pred_sana} ({100*n_pred_sana/n_total:.1f}%)")
    print(f"Predichas como 'leucemica' (esperado si aprendio dominio): {n_pred_leucemica} ({100*n_pred_leucemica/n_total:.1f}%)")

    print("\nDesglose por tipo de celula:")
    print(valid_df.groupby("cell_type")["prediccion"].value_counts())

    print("\nProbabilidad promedio de 'leucemica' asignada (0=muy seguro sana, 1=muy seguro leucemica):")
    print(f"{probs.mean():.4f}")

    if n_pred_leucemica / n_total > 0.5:
        print("\nCONCLUSION: el clasificador predice 'leucemica' para la mayoria de celulas "
              "morfologicamente normales del dominio leucemico. Esto CONFIRMA que aprendio "
              "un shortcut de dominio/adquisicion de imagen, no morfologia celular real.")
    else:
        print("\nCONCLUSION: el clasificador predice 'sana' para la mayoria de estas celulas "
              "normales, a pesar de venir del dominio de imagen leucemico. Esto es evidencia "
              "a favor de que aprendio morfologia real, no solo el origen de la imagen.")

    out_csv = args.domain_control_csv.replace(".csv", "_predicciones.csv")
    valid_df.to_csv(out_csv, index=False)
    print(f"\nDetalle completo guardado en: {out_csv}")


if __name__ == "__main__":
    main()