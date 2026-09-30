r"""
train_blip_vqa.py

Fine-tuning de BLIP en modo VQA (pregunta+imagen -> respuesta), replicando
el enfoque del repositorio original para obtener el diagnostico de forma
directa y robusta, en vez de buscar palabras clave en un caption libre
(que fallaba, por ejemplo, con "small"/"overall" conteniendo "all").

Uso (PowerShell):
    python train_blip_vqa.py `
        --vqa_csv .\data\vqa_pairs.csv `
        --output_dir .\checkpoints\hemblip_vqa
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


class VQADataset(Dataset):
    def __init__(self, df, processor, max_length=32):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        question = str(row["question"])
        answer = str(row["answer"])

        inputs = self.processor(
            images=image, text=question, padding="max_length",
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            answer, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_model(base_model_name, device):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForQuestionAnswering.from_pretrained(base_model_name)
    for p in model.vision_model.parameters():
        p.requires_grad = False
    model.to(device)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_csv", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="Salesforce/blip-vqa-base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hemblip_vqa")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    df = pd.read_csv(args.vqa_csv)
    print(f"Pares pregunta-respuesta: {len(df)}")

    model, processor = build_model(args.base_model_name, device)

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    train_loader = DataLoader(VQADataset(train_df, processor), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(VQADataset(val_df, processor), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} - train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(best_state, os.path.join(args.output_dir, "vqa_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping en epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    final_path = os.path.join(args.output_dir, "hemblip_vqa_final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Entrenamiento completo. Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
