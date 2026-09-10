r"""
train_hemblip_base.py

Replica del HemBLIP base (van Logtestijn & Manescu, 2026):
- Arquitectura: BLIP (Salesforce/blip-image-captioning-base) -> ViT-B/16 encoder + decoder Transformer
- Adaptacion: LoRA sobre las capas de atencion del decoder (self-attn y cross-attn), encoder visual CONGELADO
- Optimizador: AdamW, lr=5e-5, early stopping por validation loss
- Estrategia de entrenamiento: CURRICULUM en dos etapas
    Etapa 1: fine-tune con celulas SANAS (WBCAtt)
    Etapa 2: continuar el fine-tune (a partir de los pesos de la etapa 1) con celulas LEUCEMICAS (LeukemiaAttri)

Uso (PowerShell):
    python train_hemblip_base.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --images_root .\data\images `
        --output_dir .\checkpoints\hemblip_base

Cada CSV debe tener, como minimo, las columnas:
    image_path  -> ruta relativa (a --images_root) o absoluta de la imagen
    caption     -> descripcion morfologica en lenguaje natural

Si tu dataset esta en un solo CSV con una columna 'label' (healthy/leukemic),
usa --single_csv y --label_col en vez de --healthy_csv/--leukemic_csv (ver mas abajo).
"""

import argparse
import os
import copy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CellCaptionDataset(Dataset):
    def __init__(self, df, images_root, processor, max_length=64):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if self.images_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.images_root, img_path)
        image = Image.open(img_path).convert("RGB")
        caption = str(row["caption"])

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Quitar la dimension de batch que agrega el processor
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    # Labels = input_ids, con padding enmascarado a -100 para no penalizar el loss ahi
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# --------------------------------------------------------------------------- #
# Modelo: BLIP base + LoRA solo en el decoder, encoder visual congelado
# --------------------------------------------------------------------------- #
def build_model(base_model_name="Salesforce/blip-image-captioning-base", device="cuda"):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    # Congelar TODO el encoder visual (ViT-B/16)
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # LoRA sobre las proyecciones de atencion del decoder (self-attn y cross-attn)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "key", "value", "output.dense"],
        task_type=TaskType.CAUSAL_LM,
    )
    model.text_decoder = get_peft_model(model.text_decoder, lora_config)

    model.to(device)
    return model, processor


# --------------------------------------------------------------------------- #
# Entrenamiento con early stopping por validation loss
# --------------------------------------------------------------------------- #
def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    stage_name,
    output_dir,
    lr=5e-5,
    max_epochs=30,
    patience=3,
):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch} - train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{stage_name}] Epoch {epoch} - val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        print(f"[{stage_name}] Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(best_state, os.path.join(output_dir, f"{stage_name}_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{stage_name}] Early stopping en epoch {epoch} (sin mejora en {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model


def make_loaders(df, images_root, processor, batch_size, val_size=0.1, seed=42):
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
    train_ds = CellCaptionDataset(train_df, images_root, processor)
    val_ds = CellCaptionDataset(val_df, images_root, processor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, default=None)
    parser.add_argument("--leukemic_csv", type=str, default=None)
    parser.add_argument("--single_csv", type=str, default=None,
                         help="Alternativa: un solo CSV con columna --label_col para separar sano/leucemico")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--healthy_label", type=str, default="healthy")
    parser.add_argument("--leukemic_label", type=str, default="leukemic")
    parser.add_argument("--images_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hemblip_base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --- cargar y separar datasets sano / leucemico ---
    if args.single_csv:
        df = pd.read_csv(args.single_csv)
        healthy_df = df[df[args.label_col] == args.healthy_label]
        leukemic_df = df[df[args.label_col] == args.leukemic_label]
    else:
        assert args.healthy_csv and args.leukemic_csv, (
            "Debes dar --healthy_csv y --leukemic_csv, o bien --single_csv con --label_col."
        )
        healthy_df = pd.read_csv(args.healthy_csv)
        leukemic_df = pd.read_csv(args.leukemic_csv)

    print(f"Celulas sanas: {len(healthy_df)} | Celulas leucemicas: {len(leukemic_df)}")

    # --- construir modelo base (BLIP + LoRA en decoder, ViT congelado) ---
    model, processor = build_model(device=device)

    # --- Etapa 1: entrenar con celulas SANAS ---
    healthy_train_loader, healthy_val_loader = make_loaders(
        healthy_df, args.images_root, processor, args.batch_size
    )
    model = train_stage(
        model, healthy_train_loader, healthy_val_loader, device,
        stage_name="stage1_healthy",
        output_dir=args.output_dir,
        lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
    )

    # --- Etapa 2: continuar entrenamiento con celulas LEUCEMICAS ---
    leukemic_train_loader, leukemic_val_loader = make_loaders(
        leukemic_df, args.images_root, processor, args.batch_size
    )
    model = train_stage(
        model, leukemic_train_loader, leukemic_val_loader, device,
        stage_name="stage2_leukemic",
        output_dir=args.output_dir,
        lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
    )

    # --- fusionar LoRA en el decoder antes de guardar ---
    # Sin esto, save_pretrained() guarda los pesos con los nombres internos
    # que usa PEFT (prefijo "base_model.model..."), que no coinciden con lo
    # que espera BlipForConditionalGeneration al recargarlo normalmente.
    model.text_decoder = model.text_decoder.merge_and_unload()

    # --- guardar modelo final ---
    final_path = os.path.join(args.output_dir, "hemblip_base_final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Entrenamiento completo. Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
