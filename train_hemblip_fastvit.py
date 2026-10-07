r"""
train_hemblip_fastvit.py

Objetivo especifico 4 de la propuesta de TT: sustituir el encoder visual
(ViT-B/16) por FastViT y comparar el desempeno contra la linea base.

Como FastViT no es un encoder "plug and play" para BLIP (produce un mapa de
caracteristicas con dimension y forma distintas a las que espera el decoder),
se envuelve en FastViTVisionWrapper: corre FastViT (congelado, preentrenado
en ImageNet), aplana su mapa de caracteristicas en una secuencia de tokens,
y los proyecta (capa lineal ENTRENABLE) a la dimension que el decoder de
BLIP espera (768). Ese envoltorio reemplaza el atributo vision_model del
BlipForConditionalGeneration original, de modo que el resto del modelo
(decoder, LoRA, generate(), etc.) sigue funcionando sin cambios.

El resto de la configuracion (LoRA solo en cross-attention r=8/alpha=16,
ultimas 2 capas + LM head + embeddings entrenados completos, AdamW,
scheduler coseno, mixed precision) es IDENTICA a train_hemblip_exact_replica.py,
para que la comparacion final aisle el efecto del encoder.

Requiere: pip install timm

Uso (PowerShell):
    python train_hemblip_fastvit.py `
        --healthy_csv .\data\healthy_captions_faithful_dedup.csv `
        --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
        --output_dir .\checkpoints\hemblip_fastvit `
        --fastvit_name fastvit_sa36.apple_dist_in1k
"""

import argparse
import math
import os
import timm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model


TARGET_MODULES = [
    "crossattention.self.query",
    "crossattention.self.key",
    "crossattention.self.value",
    "crossattention.output.dense",
]
K_LAST = 2


class FastViTVisionWrapper(nn.Module):
    """Envuelve FastViT para que 'se vea' como el vision_model de BLIP:
    recibe pixel_values, devuelve un objeto con .last_hidden_state de la
    dimension que el decoder espera."""

    def __init__(self, fastvit_name, target_hidden_size, image_size=256):
        super().__init__()
        self.backbone = timm.create_model(fastvit_name, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Intentar reparametrizar a la forma de inferencia (mas rapida), si
        # el modelo lo soporta -- no afecta la salida, solo la velocidad.
        if hasattr(self.backbone, "reparameterize"):
            try:
                self.backbone.reparameterize()
            except Exception:
                pass

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_map = self.backbone.forward_features(dummy)
            in_channels = feat_map.shape[1]
            self.feat_hw = feat_map.shape[2], feat_map.shape[3]

        self.proj = nn.Linear(in_channels, target_hidden_size)  # ENTRENABLE

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # el backbone SIEMPRE en eval, congelado de verdad
        return self

    def forward(self, pixel_values, **kwargs):
        with torch.no_grad():
            feat_map = self.backbone.forward_features(pixel_values)  # (B, C, H, W)
        b, c, h, w = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.proj(tokens)  # (B, H*W, hidden_decoder)
        return BaseModelOutput(last_hidden_state=tokens)


class CellCaptionDataset(Dataset):
    """Usa el transform propio de FastViT para la imagen (distinto al de
    BlipProcessor), y el tokenizer de BlipProcessor para el texto."""

    def __init__(self, df, tokenizer, image_transform, max_length=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.image_transform(image)

        caption = str(row["caption"])
        tok = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }


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


def build_model(base_model_name, fastvit_name, device):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    tok = processor.tokenizer
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = getattr(tok, "sep_token_id", tok.eos_token_id)
    model.config.decoder_start_token_id = getattr(tok, "cls_token_id", tok.bos_token_id)

    hidden_size = model.config.text_config.hidden_size

    # --- Sustituir el encoder visual: ViT-B/16 -> FastViT ---
    model.vision_model = FastViTVisionWrapper(fastvit_name, hidden_size)

    # --- Misma config de LoRA que train_hemblip_exact_replica.py ---
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
    model.print_trainable_parameters()  # aqui deberias ver tambien los parametros de vision_model.proj

    model.to(device)

    # Transform propio de FastViT (normalizacion/tamano que espera ese backbone)
    fastvit_probe = timm.create_model(fastvit_name, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(fastvit_probe)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    del fastvit_probe

    return model, processor, image_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_csv", type=str, required=True)
    parser.add_argument("--leukemic_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hemblip_fastvit")
    parser.add_argument("--base_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--fastvit_name", type=str, default="fastvit_sa36.apple_dist_in1k")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
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

    model, processor, image_transform = build_model(args.base_model_name, args.fastvit_name, device)
    pad_token_id = processor.tokenizer.pad_token_id

    train_df, val_df = train_test_split(combined_df, test_size=args.val_size, random_state=args.seed)
    train_ds = CellCaptionDataset(train_df, processor.tokenizer, image_transform)
    val_ds = CellCaptionDataset(val_df, processor.tokenizer, image_transform)
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
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"[FastViT] Epoch {epoch} - train")
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
            for batch in tqdm(val_loader, desc=f"[FastViT] Epoch {epoch} - val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    out = model(**batch)
                val_sum += out.loss.item()
        val_loss = val_sum / len(val_loader)

        print(f"[FastViT] Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

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
    merged = model.merge_and_unload()  # solo fusiona las capas con LoRA; vision_model.proj queda igual

    final_path = os.path.join(args.output_dir, "hemblip_fastvit_final.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "state_dict": merged.state_dict(),
        "fastvit_name": args.fastvit_name,
        "base_model_name": args.base_model_name,
    }, final_path)
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    print(f"Entrenamiento completo. Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
