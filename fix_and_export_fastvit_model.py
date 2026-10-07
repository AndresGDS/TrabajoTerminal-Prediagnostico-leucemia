r"""
fix_and_export_fastvit_model.py

El entrenamiento de train_hemblip_fastvit.py completo sus 10 epocas
correctamente (val_loss convergio de 0.14 a 0.037, con 8 checkpoints
intermedios guardados sin problema). El UNICO fallo ocurrio en el guardado
final del modelo fusionado (merge_and_unload + torch.save), por un error de
escritura a disco (espacio lleno, antivirus bloqueando el archivo a medio
escribir, o similar) -- no por un error de nuestro codigo ni del
entrenamiento en si.

Este script reconstruye la arquitectura, carga el ULTIMO checkpoint
intermedio que si se guardo bien (best.pt, de la epoca con mejor val_loss),
fusiona LoRA, y reintenta el guardado final -- sin reentrenar nada.

Uso (PowerShell):
    python fix_and_export_fastvit_model.py `
        --checkpoint .\checkpoints\hemblip_fastvit\best.pt `
        --output_dir .\checkpoints\hemblip_fastvit `
        --fastvit_name fastvit_sa36.apple_dist_in1k
"""

import argparse
import os
import timm
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
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


def build_model_with_lora(base_model_name, fastvit_name, device):
    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    hidden_size = model.config.text_config.hidden_size
    model.vision_model = FastViTVisionWrapper(fastvit_name, hidden_size)

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
    model.to(device)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="El best.pt que SI se guardo bien durante el entrenamiento")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--fastvit_name", type=str, default="fastvit_sa36.apple_dist_in1k")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Reconstruyendo arquitectura (BLIP + FastViT + LoRA)...")
    model, processor = build_model_with_lora(args.base_model_name, args.fastvit_name, device)

    print(f"Cargando checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Cargado sin problemas. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("Fusionando pesos LoRA...")
    merged = model.merge_and_unload()

    final_path = os.path.join(args.output_dir, "hemblip_fastvit_final.pt")
    print(f"Guardando modelo final en: {final_path}")
    torch.save({
        "state_dict": merged.state_dict(),
        "fastvit_name": args.fastvit_name,
        "base_model_name": args.base_model_name,
    }, final_path)

    processor_dir = os.path.join(args.output_dir, "processor")
    processor.save_pretrained(processor_dir)

    print(f"\nListo. Modelo final: {final_path}")
    print(f"Processor: {processor_dir}")


if __name__ == "__main__":
    main()
