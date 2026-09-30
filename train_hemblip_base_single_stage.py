r"""
train_hemblip_base_single_stage.py

Variante fiel al metodo ORIGINAL del paper HemBLIP: entrena en UN SOLO PASO
sobre la union de las ~14k parejas imagen-texto (sanas + leucemicas
mezcladas y barajadas), en vez del curriculum en dos etapas de
train_hemblip_base.py. Misma arquitectura: BLIP + LoRA en decoder, encoder
visual (ViT-B/16) congelado, AdamW lr=5e-5, early stopping por val_loss.

Sirve para comparar contra la version de 2 etapas y ver si el curriculum
realmente ayuda o si introduce olvido catastrofico (como se observo en las
pruebas manuales del modelo de 2 etapas).

Uso (PowerShell):
    python train_hemblip_base_single_stage.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --output_dir .\checkpoints\hemblip_base_single_stage `
        --batch_size 4
"""

import argparse
import copy
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


class CellCaptionDataset(Dataset):
    def __init__(self, df, processor, max_length=64):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        caption = str(row["caption"])

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_model(base_model_name="Salesforce/blip-image-captioning-base", device="cuda"):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    for param in model.vision_model.parameters():
        param.requires_grad = False

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


def train_stage(model, train_loader, val_loader, device, output_dir,
                 lr=5e-5, max_epochs=30, patience=3):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[single_stage] Epoch {epoch} - train"):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[single_stage] Epoch {epoch} - val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        print(f"[single_stage] Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(best_state, os.path.join(output_dir, "single_stage_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[single_stage] Early stopping en epoch {epoch} (sin mejora en {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hemblip_base_single_stage")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    healthy_df = pd.read_csv(args.healthy_csv)
    leukemic_df = pd.read_csv(args.leukemic_csv)
    print(f"Celulas sanas: {len(healthy_df)} | Celulas leucemicas: {len(leukemic_df)}")

    # --- Union + mezcla, fiel al metodo original del paper (un solo dataset) ---
    combined_df = pd.concat([healthy_df, leukemic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"Total combinado y barajado: {len(combined_df)} pares imagen-caption")

    model, processor = build_model(device=device)

    train_df, val_df = train_test_split(combined_df, test_size=args.val_size, random_state=args.seed)
    train_ds = CellCaptionDataset(train_df, processor)
    val_ds = CellCaptionDataset(val_df, processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = train_stage(
        model, train_loader, val_loader, device,
        output_dir=args.output_dir,
        lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
    )

    # --- fusionar LoRA antes de guardar (mismo fix que en la version de 2 etapas) ---
    model.text_decoder = model.text_decoder.merge_and_unload()

    final_path = os.path.join(args.output_dir, "hemblip_base_single_stage_final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Entrenamiento completo. Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
