r"""
test_resolution_shortcut.py

Experimento causal sobre un posible atajo de dominio: las imagenes sanas
(WBCAtt) miden ~360x363 px y los recortes leucemicos ~61x103 px. Aqui se
toman imagenes SANAS que el modelo nunca vio (split de prueba de WBCAtt) y
se les cambia SOLO la resolucion/encuadre para parecerse a un recorte
leucemico. Si el modelo empieza a mencionar blastos o a cambiar de
plantilla de caption, esas pistas visuales estan afectando su decision.

Condiciones:
  original                        -> sin cambios (referencia)
  baja_resolucion                 -> se reduce a (ancho x alto) tipico leucemico
  recorte_centro+baja_resolucion  -> ademas, recorte apretado al centro
                                     (la celula en WBCAtt esta centrada)

Limite: el recorte al centro es una aproximacion; no reproduce exactamente
el encuadre de una bounding box. Tampoco aisla el color/tincion.

Uso (PowerShell), con la GPU libre:
    python test_resolution_shortcut.py `
        --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
        --healthy_csv .\data_ft_test\healthy_captions.csv `
        --n_samples 200 `
        --out_csv .\resolution_shortcut_results.csv
"""

import argparse
import re
from math import comb

import pandas as pd
from PIL import Image


BLAST_KEYWORDS = ["myeloblast", "lymphoblast", "monoblast",
                  "abnormal promyelocyte", "promyelocyte"]


def has_blast_word(text):
    lower = str(text).lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in BLAST_KEYWORDS)


def healthy_style(text):
    """Plantilla de las sanas de WBCAtt: 'this is a ...'."""
    return str(text).strip().lower().startswith("this is")


def lower_resolution(img, size):
    return img.resize(size, Image.LANCZOS)


def center_crop(img, frac):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def generate(model, processor, device, images, batch_size):
    import torch
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(
                pixel_values=inputs["pixel_values"], max_length=64, num_beams=4,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        out.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return out


def main():
    import torch
    from tqdm import tqdm
    from transformers import BlipProcessor, BlipForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--healthy_csv", required=True,
                        help="CSV de sanas NO usadas en entrenamiento (ej. data_ft_test\\healthy_captions.csv)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--target_width", type=int, default=61)
    parser.add_argument("--target_height", type=int, default=103)
    parser.add_argument("--center_frac", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", default="./resolution_shortcut_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.healthy_csv)
    df = df.sample(n=min(args.n_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    size = (args.target_width, args.target_height)
    conditions = {
        "original": lambda im: im,
        "baja_resolucion": lambda im: lower_resolution(im, size),
        "recorte_centro+baja_resolucion":
            lambda im: lower_resolution(center_crop(im, args.center_frac), size),
    }

    originals, valid_paths = [], []
    for p in df["image_path"]:
        try:
            originals.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception:
            continue
    print(f"Imagenes sanas cargadas: {len(originals)}")

    result = pd.DataFrame({"image_path": valid_paths})
    for name, fn in conditions.items():
        print(f"\nGenerando condicion: {name}")
        imgs = [fn(im) for im in originals]
        caps = []
        for i in tqdm(range(0, len(imgs), args.batch_size), desc=name):
            caps.extend(generate(model, processor, device,
                                 imgs[i:i + args.batch_size], args.batch_size))
        result[f"caption__{name}"] = caps
        result[f"alarma__{name}"] = [has_blast_word(c) for c in caps]
        result[f"estilo_sana__{name}"] = [healthy_style(c) for c in caps]

    result.to_csv(args.out_csv, index=False)

    n = len(result)
    print("\n" + "=" * 72)
    print("RESULTADO -- imagenes SANAS nunca vistas, con resolucion/encuadre alterados")
    print("=" * 72)
    print(f"{'condicion':34s} {'alarma blasto':>14s} {'estilo sana':>12s} {'McNemar p':>10s}")
    base_alarm = result["alarma__original"]
    for name in conditions:
        alarm = result[f"alarma__{name}"]
        style = result[f"estilo_sana__{name}"]
        if name == "original":
            p_txt = "-"
        else:
            b = int((~base_alarm & alarm).sum())   # sin alarma -> con alarma
            c = int((base_alarm & ~alarm).sum())   # con alarma -> sin alarma
            p_txt = f"{mcnemar_exact(b, c):.4f}"
        print(f"{name:34s} {100 * alarm.mean():13.1f}% {100 * style.mean():11.1f}% {p_txt:>10s}")

    print(f"\n(n = {n} imagenes sanas por condicion)")
    print("Lectura: si 'alarma blasto' sube y 'estilo sana' baja al degradar, la resolucion/"
          "encuadre influye en la decision del modelo (atajo). Si casi no cambian, "
          "el exceso de alarmas viene de otra causa.")
    print(f"Detalle guardado en: {args.out_csv}")


if __name__ == "__main__":
    main()
