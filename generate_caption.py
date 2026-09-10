r"""
generate_caption.py

Prueba rapida: carga el modelo HemBLIP base ya entrenado y genera la
descripcion morfologica de una sola imagen.

Uso (PowerShell):
    python generate_caption.py `
        --model_dir .\checkpoints\hemblip_base\hemblip_base_final `
        --image .\ruta\a\una_celula.jpg
"""

import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Carpeta del modelo final (ej. .\\checkpoints\\hemblip_base\\hemblip_base_final)")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen a describir")
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando modelo...")
    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=args.max_length,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print(f"Imagen:  {args.image}")
    print(f"Caption: {caption}")
    print("=" * 60)


if __name__ == "__main__":
    main()
