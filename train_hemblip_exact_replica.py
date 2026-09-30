r"""
train_hemblip_exact_replica.py

Replica EXACTA de la configuracion de entrenamiento encontrada en el
repositorio oficial del paper (train_blip1_lora_modules_to_save.py,
julievanlogtestijn-ucl/Vision-Language-Models-for-Leukemia-Detection),
a diferencia de train_hemblip_base_single_stage.py que usaba supuestos
razonables pero no verificados.

Diferencias clave frente a nuestra version anterior:
  - LoRA SOLO en cross-attention del decoder (no en self-attention)
  - r=8, alpha=16 (no 16/32)
  - Ademas de LoRA: las ULTIMAS 2 capas del decoder, el LM head y los
    embeddings se entrenan por fine-tuning COMPLETO (modules_to_save)
  - Batch efectivo 32 (batch_size=4 x grad_accum=8)
  - Scheduler coseno con warmup (10%)
  - Weight decay 0.05
  - Mixed precision (AMP + GradScaler)
  - max_epochs=10, patience=2 (no 30/3)

Uso (PowerShell):
    python train_hemblip_exact_replica.py `
        --healthy_csv .\data\healthy_captions.csv `
        --leukemic_csv .\data\leukemic_captions.csv `
        --output_dir .\checkpoints\hemblip_exact_replica
"""

import argparse
import math
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


# --------------------------------------------------------------------------- #
# Configuracion identica al repositorio original
# --------------------------------------------------------------------------- #
TARGET_MODULES = [
    "crossattention.self.query",
    "crossattention.self.key",
    "crossattention.self.value",
    "crossattention.output.dense",
]
K_LAST = 2  # ultimas 2 capas del decoder, entrenadas completas (no LoRA)


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
            images=image, text=caption, padding="max_length",
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def collate_fn(batch, pad_token_id):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_model(base_model_name, device):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    tok = processor.tokenizer
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = getattr(tok, "sep_token_id", tok.eos_token_id)
    model.config.decoder_start_token_id = getattr(tok, "cls_token_id", tok.bos_token_id)

    # Congelar encoder visual
    for p in model.vision_model.parameters():
        p.requires_grad = False

    # modules_to_save: LM head + embeddings + ultimas K_LAST capas del decoder,
    # entrenadas COMPLETAS (no via LoRA)
    modules_to_save = [
        "text_decoder.cls",
        "text_decoder.bert.embeddings.word_embeddings",
        "text_decoder.bert.embeddings.position_embeddings",
        "text_decoder.bert.embeddings.LayerNorm",
    ] + [f"text_decoder.bert.encoder.layer.{i}" for i in range(12 - K_LAST, 12)]

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hemblip_exact_replica")
    parser.add_argument("--base_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)  # batch efectivo = batch_size * grad_accum = 32
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    torch.manual_seed(args.seed)

    healthy_df = pd.read_csv(args.healthy_csv)
    leukemic_df = pd.read_csv(args.leukemic_csv)
    combined_df = pd.concat([healthy_df, leukemic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"Total combinado y barajado: {len(combined_df)} pares imagen-caption")

    model, processor = build_model(args.base_model_name, device)
    pad_token_id = processor.tokenizer.pad_token_id

    train_df, val_df = train_test_split(combined_df, test_size=args.val_size, random_state=args.seed)
    train_ds = CellCaptionDataset(train_df, processor)
    val_ds = CellCaptionDataset(val_df, processor)
    collate = lambda b: collate_fn(b, pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda") if device == "cuda" else GradScaler(enabled=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch} - train")
        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.3f}")
        train_loss = running / len(train_loader)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    out = model(**batch)
                val_sum += out.loss.item()
        val_loss = val_sum / len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
            print(f"Checkpoint guardado (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping en epoch {epoch}.")
                break

    model.load_state_dict(best_state)

    # Fusionar LoRA (los modules_to_save ya quedan como pesos normales, no
    # necesitan fusion -- merge_and_unload solo afecta a las capas con LoRA)
    merged = model.merge_and_unload()

    final_path = os.path.join(args.output_dir, "hemblip_exact_replica_final")
    os.makedirs(final_path, exist_ok=True)
    merged.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Entrenamiento completo. Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
