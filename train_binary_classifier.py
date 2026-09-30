r"""
train_binary_classifier.py

Version 2 del diagnostico: en vez de buscar palabras clave en el texto
generado (regex), esto entrena un clasificador REAL sobre los embeddings
visuales que ya extrae el encoder congelado (ViT-B/16) de tu modelo
HemVLM. Es el mismo enfoque de la seccion 3.3 del paper HemBLIP
("HemBLIP as a feature extractor").

Como el encoder visual esta congelado, sus embeddings son los mismos sin
importar que tan entrenado este el decoder de texto -- por eso este
clasificador es independiente de la calidad del captioning.

Pasos:
  1. Carga el modelo entrenado (solo se usa su encoder visual).
  2. Extrae un embedding por imagen (mean-pooling sobre los patches del ViT)
     para TODAS las imagenes de healthy_captions.csv y leukemic_captions.csv.
  3. Entrena una regresion logistica (sana=0, leucemica=1) sobre esos
     embeddings, con split train/test.
  4. Imprime accuracy, F1 y matriz de confusion.
  5. Guarda el clasificador entrenado en un .joblib, listo para usarse en
     api.py.

Uso (PowerShell):
    python train_binary_classifier.py `
        --model_dir .\checkpoints\hemblip_base_single_stage_v2\hemblip_base_single_stage_final `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --out_path .\checkpoints\binary_classifier.joblib
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_vision_encoder(model_dir, device):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return model.vision_model, processor, model


@torch.no_grad()
def extract_embedding(vision_model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = vision_model(pixel_values=inputs["pixel_values"])
    # mean-pooling sobre todos los patches (incluye el token [CLS]-like de BLIP)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


def build_dataset(healthy_csv, leukemic_csv, vision_model, processor, device):
    healthy_df = pd.read_csv(healthy_csv)
    leukemic_df = pd.read_csv(leukemic_csv)

    healthy_df["label"] = 0  # sana
    leukemic_df["label"] = 1  # leucemica

    combined = pd.concat([healthy_df[["image_path", "label"]],
                           leukemic_df[["image_path", "label"]]], ignore_index=True)

    embeddings = []
    labels = []
    skipped = 0
    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Extrayendo embeddings"):
        try:
            emb = extract_embedding(vision_model, processor, row["image_path"], device)
            embeddings.append(emb)
            labels.append(row["label"])
        except Exception:
            skipped += 1
            continue

    print(f"Embeddings extraidos: {len(embeddings)}. Omitidas (imagen invalida): {skipped}")
    return np.array(embeddings), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Carpeta del modelo entrenado (solo se usa su encoder visual)")
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="./checkpoints/binary_classifier.joblib")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando encoder visual del modelo...")
    vision_model, processor, _ = load_vision_encoder(args.model_dir, device)

    print("Construyendo dataset de embeddings (esto puede tardar unos minutos)...")
    X, y = build_dataset(args.healthy_csv, args.leukemic_csv, vision_model, processor, device)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("Entrenando regresion logistica...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL CLASIFICADOR BINARIO (sana=0, leucemica=1)")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=["sana", "leucemica"]))

    joblib.dump(clf, args.out_path)
    print(f"\nClasificador guardado en: {args.out_path}")


if __name__ == "__main__":
    main()