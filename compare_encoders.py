r"""
compare_encoders.py

Cierra el objetivo especifico 4 de la propuesta de TT: compara la linea
base (ViT-B/16) contra la variante con FastViT, en dos ejes:

  1. Calidad descriptiva: BLEU, ROUGE-1/L, BERTScore (BioBERT) sobre el
     mismo set de prueba oficial no visto (leukemic_captions_faithful_test.csv).
  2. Robustez al sesgo de dominio: reutiliza el mismo control de dominio
     (celulas normales dentro de imagenes leucemicas) para ver si el
     encoder nuevo generaliza mejor o igual de mal que el original.

Uso (PowerShell):
    python compare_encoders.py `
        --baseline_model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
        --fastvit_checkpoint .\checkpoints\hemblip_fastvit\hemblip_fastvit_final.pt `
        --fastvit_processor_dir .\checkpoints\hemblip_fastvit\processor `
        --test_csv .\data\leukemic_captions_faithful_test.csv `
        --out_csv .\comparacion_encoders.csv
"""

import argparse
import timm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


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


def load_baseline(model_dir, device):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    return model, processor, preprocess


def load_fastvit(checkpoint_path, processor_dir, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BlipForConditionalGeneration.from_pretrained(ckpt["base_model_name"])
    hidden_size = model.config.text_config.hidden_size
    model.vision_model = FastViTVisionWrapper(ckpt["fastvit_name"], hidden_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    processor = BlipProcessor.from_pretrained(processor_dir)
    fastvit_probe = timm.create_model(ckpt["fastvit_name"], pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(fastvit_probe)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    del fastvit_probe

    def preprocess(image):
        return image_transform(image)

    return model, processor, preprocess


def generate_captions(model, processor, preprocess, image_paths, device):
    captions = []
    for path in tqdm(image_paths, desc="Generando captions"):
        image = Image.open(path).convert("RGB")
        pixel_values = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values, max_length=64, num_beams=4,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        captions.append(processor.decode(output_ids[0], skip_special_tokens=True))
    return captions


def compute_metrics(hypotheses, references, device):
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
        for ref, hyp in zip(references, hypotheses)
    ]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores, rougeL_scores = [], []
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        rouge1_scores.append(s["rouge1"].fmeasure)
        rougeL_scores.append(s["rougeL"].fmeasure)
    _, _, F1 = bert_score(
        cands=hypotheses, refs=references, lang="en",
        model_type="./biobert_local", num_layers=12, idf=False, device=device,
    )
    return {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "BERTScore-F1": float(F1.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_dir", type=str, required=True)
    parser.add_argument("--fastvit_checkpoint", type=str, required=True)
    parser.add_argument("--fastvit_processor_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="./comparacion_encoders.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    df = pd.read_csv(args.test_csv)
    references = df["caption"].astype(str).tolist()
    image_paths = df["image_path"].tolist()

    print("\n--- Evaluando linea base (ViT-B/16) ---")
    model_b, proc_b, prep_b = load_baseline(args.baseline_model_dir, device)
    hyps_baseline = generate_captions(model_b, proc_b, prep_b, image_paths, device)
    metrics_baseline = compute_metrics(hyps_baseline, references, device)
    del model_b
    torch.cuda.empty_cache()

    print("\n--- Evaluando variante FastViT ---")
    model_f, proc_f, prep_f = load_fastvit(args.fastvit_checkpoint, args.fastvit_processor_dir, device)
    hyps_fastvit = generate_captions(model_f, proc_f, prep_f, image_paths, device)
    metrics_fastvit = compute_metrics(hyps_fastvit, references, device)

    print("\n" + "=" * 70)
    print("COMPARACION FINAL -- ViT-B/16 (linea base) vs FastViT")
    print("=" * 70)
    comparison = pd.DataFrame([
        {"Encoder": "ViT-B/16 (linea base)", **metrics_baseline},
        {"Encoder": "FastViT", **metrics_fastvit},
    ])
    print(comparison.to_string(index=False))

    comparison.to_csv(args.out_csv, index=False)
    print(f"\nGuardado en: {args.out_csv}")

    out_captions = pd.DataFrame({
        "image_path": image_paths,
        "reference": references,
        "caption_vitb16": hyps_baseline,
        "caption_fastvit": hyps_fastvit,
    })
    out_captions.to_csv(args.out_csv.replace(".csv", "_captions_detalle.csv"), index=False)
    print(f"Detalle de captions por imagen guardado en: {args.out_csv.replace('.csv', '_captions_detalle.csv')}")


if __name__ == "__main__":
    main()
