r"""
train_classifier_dinobloom.py

Repite el experimento de train_binary_classifier.py + validate_domain_shortcut.py
pero usando DinoBloom en vez del encoder de tu BLIP entrenado. DinoBloom es
DINOv2 preentrenado especificamente sobre >380,000 imagenes de celulas de
sangre y medula osea (13 datasets distintos) -- exactamente el tipo de
encoder que podria no caer en el shortcut de dominio que detectamos,
porque ya "vio" diversidad real de adquisicion de imagen hematologica
durante su propio preentrenamiento.

Requiere descargar los pesos de DinoBloom (Apache 2.0, sin restriccion de
acceso) la primera vez que se corre -- se descargan solos desde HuggingFace.

Uso (PowerShell):
    python train_classifier_dinobloom.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --domain_control_csv .\data\domain_control.csv `
        --variant b `
        --out_path .\checkpoints\dinobloom_classifier.joblib
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm


VARIANT_CONFIG = {
    "s": ("dinov2_vits14", 384),
    "b": ("dinov2_vitb14", 768),
    "l": ("dinov2_vitl14", 1024),
    "g": ("dinov2_vitg14", 1536),
}


def load_dinobloom(variant, device):
    dinov2_model_name, embed_dim = VARIANT_CONFIG[variant]
    print(f"Cargando DINOv2 base ({dinov2_model_name})...")
    model = torch.hub.load("facebookresearch/dinov2", dinov2_model_name)

    print(f"Descargando pesos de DinoBloom-{variant.upper()}...")
    ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename=f"pytorch_model_{variant}.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    num_tokens = int(1 + (224 / 14) ** 2)
    model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
def extract_embedding(model, transform, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    features = model(tensor)  # (1, embed_dim) -- CLS token pooled
    return features.squeeze(0).cpu().numpy()


def extract_from_csv(csv_path, model, transform, device, label=None, path_col="image_path"):
    df = pd.read_csv(csv_path)
    embeddings, labels, meta_rows = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embeddings de {csv_path}"):
        try:
            emb = extract_embedding(model, transform, row[path_col], device)
            embeddings.append(emb)
            labels.append(label if label is not None else row.get("label"))
            meta_rows.append(row)
        except Exception:
            continue
    return np.array(embeddings), np.array(labels), pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--domain_control_csv", type=str, default=None,
                         help="Opcional: si se da, tambien prueba contra el control de dominio")
    parser.add_argument("--variant", type=str, default="b", choices=["s", "b", "l", "g"])
    parser.add_argument("--out_path", type=str, default="./checkpoints/dinobloom_classifier.joblib")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model, transform = load_dinobloom(args.variant, device)

    X_healthy, _, _ = extract_from_csv(args.healthy_csv, model, transform, device, label=0)
    X_leukemic, _, _ = extract_from_csv(args.leukemic_csv, model, transform, device, label=1)

    X = np.concatenate([X_healthy, X_leukemic], axis=0)
    y = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_leukemic))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n" + "=" * 60)
    print(f"RESULTADOS -- DinoBloom-{args.variant.upper()} + regresion logistica")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["sana", "leucemica"]))

    joblib.dump(clf, args.out_path)
    print(f"Clasificador guardado en: {args.out_path}")

    if args.domain_control_csv:
        print("\n" + "=" * 60)
        print("VALIDACION DE CONTROL DE DOMINIO (celulas normales en imagenes leucemicas)")
        print("=" * 60)
        X_control, _, control_df = extract_from_csv(args.domain_control_csv, model, transform, device)
        preds = clf.predict(X_control)
        n_total = len(preds)
        n_leucemica = int((preds == 1).sum())
        print(f"Total evaluadas: {n_total}")
        print(f"Predichas como 'sana' (esperado si aprendio morfologia): {n_total - n_leucemica} "
              f"({100 * (n_total - n_leucemica) / n_total:.1f}%)")
        print(f"Predichas como 'leucemica' (esperado si aprendio dominio): {n_leucemica} "
              f"({100 * n_leucemica / n_total:.1f}%)")
        if n_leucemica / n_total > 0.5:
            print("\nCONCLUSION: DinoBloom TAMBIEN cae en el shortcut de dominio en tu setup.")
        else:
            print("\nCONCLUSION: DinoBloom generaliza mejor que tu encoder BLIP en esta prueba "
                  "-- evidencia de que aprendio morfologia real, no solo procedencia de imagen.")


if __name__ == "__main__":
    main()
