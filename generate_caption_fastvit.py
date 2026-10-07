r"""
generate_caption_fastvit.py

Igual que generate_caption.py, pero para el modelo con el encoder
sustituido por FastViT. Como esta arquitectura no es un
BlipForConditionalGeneration estandar (tiene el vision_model reemplazado),
no se puede cargar con from_pretrained() normal -- hay que reconstruir la
misma arquitectura y cargar el state_dict guardado por
train_hemblip_fastvit.py.

Uso (PowerShell):
    python generate_caption_fastvit.py `
        --checkpoint .\checkpoints\hemblip_fastvit\hemblip_fastvit_final.pt `
        --processor_dir .\checkpoints\hemblip_fastvit\processor `
        --image .\ruta\a\una_celula.jpg
"""

import argparse
import timm
import torch
import torch.nn as nn
from PIL import Image
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


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_model_name = ckpt["base_model_name"]
    fastvit_name = ckpt["fastvit_name"]

    model = BlipForConditionalGeneration.from_pretrained(base_model_name)
    hidden_size = model.config.text_config.hidden_size
    model.vision_model = FastViTVisionWrapper(fastvit_name, hidden_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando modelo (BLIP + FastViT)...")
    model = load_model(args.checkpoint, device)
    processor = BlipProcessor.from_pretrained(args.processor_dir)

    # Transform propio de FastViT para el preprocesamiento de imagen
    fastvit_name = torch.load(args.checkpoint, map_location="cpu")["fastvit_name"]
    fastvit_probe = timm.create_model(fastvit_name, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(fastvit_probe)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    del fastvit_probe

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values, max_length=args.max_length, num_beams=4,
            repetition_penalty=1.5, no_repeat_ngram_size=3,
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print(f"Imagen:  {args.image}")
    print(f"Caption: {caption}")
    print("=" * 60)


if __name__ == "__main__":
    main()
