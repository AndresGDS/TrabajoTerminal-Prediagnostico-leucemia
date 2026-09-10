r"""
fix_and_export_model.py

Repara el guardado final del modelo HemBLIP base ya entrenado. El bug: al
aplicar LoRA sobre el decoder, PEFT renombra internamente los pesos
(prefijo "base_model.model..."), y save_pretrained() los guardo tal cual,
lo que impide recargarlos despues como un BlipForConditionalGeneration
normal (el decoder terminaba con pesos aleatorios).

Esto NO requiere reentrenar: los checkpoints .pt guardados durante el
entrenamiento (stage1_healthy_best.pt, stage2_leukemic_best.pt) SI tienen
los pesos correctos, guardados con torch.save() normal. Este script:

  1. Reconstruye la arquitectura (BLIP + LoRA en decoder, igual que en el
     entrenamiento).
  2. Carga el checkpoint .pt indicado (por defecto, el de la ultima etapa).
  3. Fusiona los pesos LoRA en el decoder (merge_and_unload).
  4. Guarda el modelo ya fusionado en formato estandar de HuggingFace,
     listo para cargarse con BlipForConditionalGeneration.from_pretrained()
     sin depender de PEFT.

Uso (PowerShell):
    python fix_and_export_model.py `
        --checkpoint .\checkpoints\hemblip_base\stage2_leukemic_best.pt `
        --out_dir .\checkpoints\hemblip_base\hemblip_base_final_fixed
"""

import argparse
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def build_model_with_lora(base_model_name="Salesforce/blip-image-captioning-base"):
    """Debe ser IDENTICA a build_model() en train_hemblip_base.py para que
    las claves del checkpoint coincidan al cargar el state_dict."""
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
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Ruta al .pt guardado durante el entrenamiento (ej. stage2_leukemic_best.pt)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print("Reconstruyendo arquitectura (BLIP + LoRA en decoder)...")
    model, processor = build_model_with_lora(args.base_model_name)
    model.to(device)

    print(f"Cargando checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Cargado sin problemas. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("Fusionando pesos LoRA en el decoder (merge_and_unload)...")
    model.text_decoder = model.text_decoder.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Guardando modelo reparado en: {args.out_dir}")
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    print("Listo. Prueba ahora con generate_caption.py apuntando a esta carpeta.")


if __name__ == "__main__":
    main()
