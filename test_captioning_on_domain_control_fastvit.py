r"""
test_captioning_on_domain_control_fastvit.py

Igual que test_captioning_on_domain_control.py, pero para el modelo con
encoder FastViT (arquitectura personalizada, no un BlipForConditionalGeneration
estandar). Prueba el modelo de CAPTIONING real (no un clasificador aparte)
contra el conjunto de control de dominio, para comparar directamente contra
el 81.8% que dio ViT-B/16 en la misma prueba.

Uso (PowerShell):
    python test_captioning_on_domain_control_fastvit.py `
        --checkpoint .\checkpoints\hemblip_fastvit\hemblip_fastvit_final.pt `
        --processor_dir .\checkpoints\hemblip_fastvit\processor `
        --domain_control_csv .\data\domain_control_test.csv `
        --out_csv .\captioning_domain_control_results_fastvit.csv
"""

import argparse
import re
import timm
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast", "abnormal promyelocyte", "promyelocyte"]


def detect_diagnosis_from_celltype(caption_en: str) -> str:
    lower = caption_en.lower()
    for keyword in BLAST_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return "leucemica"
    return "sana"


class FastViTVisionWrapper(nn.Module):
    def __init__(self, fastvit_name, target_hidden_size, image_size=256):
        super().__init__()
        self.backbone = timm.create_model(fastvit_name, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        if hasattr(self.backbone, "reparameterize"):
            try:
                self.backbone.reparameterize()
            except Exception:
                pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_map = self.backbone.forward_features(dummy)
            in_channels = feat_map.shape[1]
        self.proj = nn.Linear(in_channels, target_hidden_size)

    def forward(self, pixel_values, **kwargs):
        with torch.no_grad():
            feat_map = self.backbone.forward_features(pixel_values)
        b, c, h, w = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        return BaseModelOutput(last_hidden_state=tokens)


def load_fastvit_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BlipForConditionalGeneration.from_pretrained(ckpt["base_model_name"])
    hidden_size = model.config.text_config.hidden_size
    model.vision_model = FastViTVisionWrapper(ckpt["fastvit_name"], hidden_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    fastvit_probe = timm.create_model(ckpt["fastvit_name"], pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(fastvit_probe)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    del fastvit_probe

    return model, image_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, required=True)
    parser.add_argument("--domain_control_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="./captioning_domain_control_results_fastvit.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando modelo FastViT...")
    model, image_transform = load_fastvit_model(args.checkpoint, device)
    processor = BlipProcessor.from_pretrained(args.processor_dir)

    df = pd.read_csv(args.domain_control_csv)
    print(f"Celulas de control a evaluar: {len(df)}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando captions"):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue

        pixel_values = image_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values, max_length=64, num_beams=4,
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
    n_correct = (out_df["prediccion"] == "sana").sum()

    print("\n" + "=" * 70)
    print("RESULTADO -- MODELO DE CAPTIONING CON FASTVIT sobre control de dominio")
    print("=" * 70)
    print(f"Total evaluadas: {n_total}")
    print(f"Correctamente descritas como 'sana': {n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"Incorrectamente mencionando un tipo de blasto: {n_total - n_correct} "
          f"({100*(n_total-n_correct)/n_total:.1f}%)")
    print(f"\nCompara contra el 81.8% de ViT-B/16 en la misma prueba "
          f"(captioning_domain_control_results_baseline.csv).")
    print(f"\nDetalle guardado en: {args.out_csv}")


if __name__ == "__main__":
    main()
