r"""
detect_and_caption_cellpose.py

Version con Cellpose (sin entrenamiento -- modelo preentrenado de uso
libre para proyectos academicos, ver nota de licencia mas abajo).

A diferencia de la version YOLO, esto no requiere entrenar nada: Cellpose
ya viene preentrenado como segmentador generalista de celulas. Ademas,
en vez de solo una caja rectangular, dibuja el CONTORNO REAL de cada
celula -- mas preciso para mostrar exactamente que region se analizo.

Requiere: pip install cellpose opencv-python scikit-image

Uso (PowerShell):
    python detect_and_caption_cellpose.py `
        --caption_model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
        --image .\ruta\a\una_imagen_de_campo_completo.png `
        --out_image .\resultado_delimitado.png

Nota de licencia: el codigo de Cellpose es BSD-3-Clause (libre). Los pesos
preentrenados del modelo generalista fueron entrenados con datos bajo
licencia CC-BY-NC (no comercial) -- para un proyecto academico como un TT
esto no representa ninguna restriccion, ya que CC-BY-NC solo limita uso
COMERCIAL.
"""

import argparse
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops
from cellpose import models as cellpose_models
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_caption_model(model_dir, device):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor


def generate_caption(model, processor, device, image_crop):
    inputs = processor(images=image_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs["pixel_values"], max_length=64, num_beams=4,
            repetition_penalty=1.5, no_repeat_ngram_size=3,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_model_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_image", type=str, default="./resultado_delimitado.png")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Diametro esperado de celula en pixeles (None = Cellpose lo estima solo)")
    parser.add_argument("--padding", type=int, default=5,
                         help="Pixeles extra de margen al recortar cada celula")
    parser.add_argument("--min_area", type=int, default=50,
                         help="Descarta regiones detectadas mas chicas que esto (ruido/artefactos)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Cargando Cellpose (descarga el modelo la primera vez que se usa)...")
    seg_model = cellpose_models.CellposeModel(gpu=(device == "cuda"))

    print("Cargando modelo de descripcion (BLIP)...")
    caption_model, processor = load_caption_model(args.caption_model_dir, device)

    print(f"Segmentando celulas en: {args.image}")
    image = Image.open(args.image).convert("RGB")
    img_array = np.array(image)

    masks, flows, styles = seg_model.eval(img_array, diameter=args.diameter)

    regions = [r for r in regionprops(masks) if r.area >= args.min_area]
    print(f"\nCelulas detectadas: {len(regions)}")
    print("=" * 70)

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, region in enumerate(regions):
        # Bounding box de la region (para recortar con margen)
        min_row, min_col, max_row, max_col = region.bbox
        pad = args.padding
        left = max(0, min_col - pad)
        top = max(0, min_row - pad)
        right = min(image.width, max_col + pad)
        bottom = min(image.height, max_row + pad)
        crop = image.crop((left, top, right, bottom))

        caption = generate_caption(caption_model, processor, device, crop)

        print(f"\nCelula #{i + 1} (area: {region.area}px)")
        print(f"  Caja: ({left}, {top}) -> ({right}, {bottom})")
        print(f"  Descripcion generada: {caption}")

        # Dibujar el CONTORNO REAL de la celula (no solo la caja), usando
        # la mascara de segmentacion de Cellpose para esta region.
        cell_mask = (masks == region.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [tuple(pt[0]) for pt in contour]
            if len(points) > 2:
                draw.line(points + [points[0]], fill="red", width=2)

        draw.text((left, max(0, top - 20)), f"#{i + 1}", fill="red", font=font)

    draw_image.save(args.out_image)
    print(f"\nImagen con delimitaciones guardada en: {args.out_image}")


if __name__ == "__main__":
    main()
