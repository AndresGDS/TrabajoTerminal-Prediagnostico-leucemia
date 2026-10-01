# HemVLM — Prediagnóstico de leucemia

Modelo de lenguaje y visión (BLIP + LoRA) para generar descripciones morfológicas
de células de sangre periférica y apoyar el prediagnóstico de leucemia.
Trabajo Terminal 2027-A140, ESCOM — IPN.

Este README cubre el pipeline **vigente**, de principio a fin: desde descargar
los datasets crudos hasta tener la interfaz web corriendo. Los comandos están
en PowerShell (Windows).

---

## 0. Preparación del entorno

```powershell
cd C:\Users\gonza\Desktop\TT
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# PyTorch con soporte CUDA -- el pip install de arriba puede instalar una
# build CPU-only por defecto. Verifica y, si hace falta, reinstala:
python -c "import torch; print(torch.cuda.is_available())"
```

Si el resultado es `False`:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# si da error de "no matching distribution", prueba con cu128 en vez de cu124
```

**Cuidado:** volver a correr `pip install -r requirements.txt` después de esto
puede reinstalar la build CPU-only de torch otra vez. Si en algún momento un
script te dice `Usando dispositivo: cpu` sin que lo esperaras, repite este
paso de verificación antes de seguir.

---

## 1. Descargar los datasets crudos (no incluidos en el repo)

### 1.1 Células sanas — atributos (WBCAtt)

- Fuente: https://github.com/apple2373/wbcatt (carpeta `submission/`)
- Descarga: `pbc_attr_v1_train.csv`, `pbc_attr_v1_val.csv`, `pbc_attr_v1_test.csv`
- Colócalos en la raíz del proyecto.

### 1.2 Células sanas — imágenes (dataset PBC)

- Fuente: Kaggle `unclesamulus/blood-cells-image-dataset` (origen académico:
  Mendeley Data, Acevedo et al.)
- Coloca las carpetas `basophil/`, `eosinophil/`, `lymphocyte/`, `monocyte/`,
  `neutrophil/` dentro de `.\bloodcells_dataset\`.

### 1.3 Células leucémicas (LeukemiaAttri)

- Repositorio: https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024
- Descarga (Google Drive, 5 archivos `.zip`): el link está en el README de ese repo.
- **Extrae los 5 zips en el MISMO destino**, para reconstruir el árbol completo:

```powershell
mkdir .\LeukemiaAttri
Get-ChildItem "$env:USERPROFILE\Downloads\LeukemiaAttri_Dataset-*.zip" | ForEach-Object {
    Expand-Archive -Path $_.FullName -DestinationPath .\LeukemiaAttri
}
```

- Usa únicamente el subconjunto **`H_100X_C2`** (el único anotado directamente
  por hematólogos; los demás tienen atributos transferidos automáticamente).

### 1.4 Verificar que la descarga quedó completa

```powershell
python verify_leukemic_images.py `
    --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
    --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train

python verify_leukemic_images.py `
    --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\test.json `
    --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\test
```

Si reporta imágenes faltantes, vuelve a descargar y extraer antes de continuar.

---

## 2. Construir el dataset de captions

### 2.1 Sanas

```powershell
python build_captions.py `
    --wbcatt_csv .\pbc_attr_v1_train.csv `
    --wbcatt_images_dir .\bloodcells_dataset `
    --out_dir .\data_ft_train
```

Esto genera `.\data_ft_train\healthy_captions.csv` — es el que se usa para
**entrenar**. (Los splits `val`/`test` de WBCAtt no se usan en el
entrenamiento, para no perder un conjunto de comparación independiente.)

### 2.2 Leucémicas (alcance fiel: 5 subtipos, todos los tipos de célula)

```powershell
python build_leukemic_captions_faithful.py `
    --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
    --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train `
    --crops_dir .\data\leukemic_crops_faithful `
    --out_csv .\data\leukemic_captions_faithful_train.csv

python build_leukemic_captions_faithful.py `
    --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\test.json `
    --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\test `
    --crops_dir .\data\leukemic_crops_faithful `
    --out_csv .\data\leukemic_captions_faithful_test.csv
```

**Importante:** `_train.csv` y `_test.csv` se usan por separado — el de train
para entrenar, el de test se reserva intacto para la evaluación. **No los
combines en un solo archivo para entrenar** (eso causa fuga de datos, ver
sección 5).

### 2.3 (Opcional, recomendado) Verificar calidad y duplicados

```powershell
python eda_full_dataset.py `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --out_dir .\eda_full_report

python dedup_and_check_sizes.py `
    --captions_csv .\data_ft_train\healthy_captions.csv `
    --details_csv .\eda_full_report\sanas_image_details.csv `
    --out_csv .\data_ft_train\healthy_captions_dedup.csv

python check_train_test_leakage.py `
    --train_csv .\data\leukemic_captions_faithful_train.csv `
    --test_csv .\data\leukemic_captions_faithful_test.csv `
    --details_csv .\eda_full_report\leucemicas_image_details.csv
```

---

## 3. Entrenar el modelo (línea base fiel al repositorio original)

```powershell
python train_hemblip_exact_replica.py `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --output_dir .\checkpoints\hemblip_exact_replica
```

Arquitectura: BLIP (`Salesforce/blip-image-captioning-base`) con encoder
visual (ViT-B/16) congelado, LoRA en cross-attention (r=8, alpha=16), y las
últimas 2 capas del decoder + LM head + embeddings entrenados completos —
configuración verificada contra el repositorio de código del estudio de
referencia. Tiempo estimado: varias horas (depende del tamaño final del
dataset tras la limpieza).

El modelo final queda en:
`.\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final\`

---

## 4. Probar el modelo con una imagen suelta

```powershell
python generate_caption.py `
    --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --image .\bloodcells_dataset\neutrophil\BNE_7323.jpg
```

---

## 5. Evaluación cuantitativa (BLEU / ROUGE / BERTScore)

```powershell
pip install sentencepiece protobuf   # necesario para el tokenizer de BioBERT

python evaluate_captions_exact.py `
    --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --test_csv .\data\leukemic_captions_faithful_test.csv `
    --out_csv .\evaluation_results.csv
```

Si BERTScore falla y ya tienes las descripciones generadas guardadas, no hace
falta repetir la generación (puede tardar bastante):

```powershell
python compute_bertscore_only.py --csv .\evaluation_results.csv
```

**Referencia del paper original:** BLEU ~0.27-0.31, ROUGE-L ~0.49-0.52,
BERTScore ~0.86-0.87. Si tus números salen muy por encima de ese rango,
es señal de sobreajuste (ver sección "Diagnóstico de sobreajuste" más abajo).

---

## 6. Validación de sesgo de dominio (recomendado antes de confiar en el diagnóstico)

```powershell
python extract_domain_control_cells.py `
    --json .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\json_labels\train.json `
    --images_dir .\LeukemiaAttri\LeukemiaAttri_Dataset\H_100X_C2\Images\train `
    --crops_dir .\data\domain_control_crops `
    --out_csv .\data\domain_control_train.csv

python train_binary_classifier.py `
    --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --out_path .\checkpoints\binary_classifier.joblib

python validate_domain_shortcut.py `
    --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --classifier_path .\checkpoints\binary_classifier.joblib `
    --domain_control_csv .\data\domain_control_train.csv
```

Si el clasificador predice "leucémica" para la mayoría de estas células
(que sabemos que son morfológicamente normales), confirma sesgo de dominio
— documentado como limitación conocida de este proyecto.

---

## 7. (Opcional) Validación externa con un dataset nunca visto

```powershell
python external_validation.py `
    --model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --images_dir .\ruta\al\dataset_externo `
    --out_csv .\validacion_externa.csv
```

---

## 8. (Opcional) Variante con encoder FastViT

```powershell
pip install timm

python train_hemblip_fastvit.py `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --output_dir .\checkpoints\hemblip_fastvit

python compare_encoders.py `
    --baseline_model_dir .\checkpoints\hemblip_exact_replica\hemblip_exact_replica_final `
    --fastvit_checkpoint .\checkpoints\hemblip_fastvit\hemblip_fastvit_final.pt `
    --fastvit_processor_dir .\checkpoints\hemblip_fastvit\processor `
    --test_csv .\data\leukemic_captions_faithful_test.csv `
    --out_csv .\comparacion_encoders.csv
```

---

## 9. (Opcional) Tercer dominio de adquisición — AML-Cytomorphology_LMU

Dataset adicional (TCIA, Matek et al.) con células de pacientes con AML **y**
de pacientes control, capturadas con el mismo microscopio — útil para intentar
romper el sesgo de dominio detectado (ver sección 6), ya que aporta una
tercera fuente de adquisición a **ambas** clases (sana/leucémica) a la vez.

### 10.1 Descargar

- Página de la colección: https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
- Descarga vía NBIA Data Retriever o descarga directa, según lo que ofrezca
  la página. La carpeta descargada normalmente trae el prefijo `PKG-`:
  `PKG-AML-Cytomorphology_LMU/`, con una subcarpeta por tipo de célula
  (`BAS/`, `EOS/`, `LYT/`, `MYO/`, etc.) y un archivo `.sums` de checksums.
- Verifica el tamaño tras extraer (deberían ser ~11.7 GB, 18,366 archivos):

```powershell
Get-ChildItem -Recurse .\PKG-AML-Cytomorphology_LMU -File | Measure-Object -Property Length -Sum
```

### 10.2 Construir captions (simples, sin atributos morfológicos detallados)

```powershell
python build_aml_cytomorphology_captions.py `
    --root_dir .\PKG-AML-Cytomorphology_LMU `
    --crops_dir .\data\aml_cytomorphology_crops `
    --out_csv .\data\aml_cytomorphology_captions.csv
```

El diagnóstico se determina por tipo de célula: `MYO`/`MOB`/`PMO`/`PMB`
(blastos/promielocitos) = `leukemia`; el resto = `healthy`.

### 10.3 Mezclar una muestra balanceada en el entrenamiento

**No uses el dataset completo** — dominaría el estilo de tus captions (mucho
más simples que los de WBCAtt/LeukemiaAttri). Se recomienda una muestra:

```powershell
python mix_third_domain.py `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --aml_cyto_csv .\data\aml_cytomorphology_captions.csv `
    --out_healthy_csv .\data\healthy_captions_augmented.csv `
    --out_leukemic_csv .\data\leukemic_captions_augmented.csv `
    --n_healthy_sample 3000 `
    --n_leukemia_sample 3000
```

Luego entrena normal, apuntando a los CSV `_augmented`:

```powershell
python train_hemblip_exact_replica.py `
    --healthy_csv .\data\healthy_captions_augmented.csv `
    --leukemic_csv .\data\leukemic_captions_augmented.csv `
    --output_dir .\checkpoints\hemblip_augmented_domain
```

**Resultado observado:** en nuestras pruebas, esta mezcla NO redujo el sesgo
de dominio medido con `validate_domain_shortcut.py` (sigue en ~99.9-100%,
incluso cambiando el encoder a DinoBloom) — evidencia de que el atajo vive en
el encoder visual congelado, no se resuelve solo agregando datos al decoder.
Ver sección "Problemas comunes" y la discusión de arquitectura en el reporte
técnico para las alternativas (normalización de tinción, entrenamiento
adversarial de dominio).

---

## 10. Desplegar la interfaz web

Dos procesos en dos terminales:

```powershell
# Terminal 1 -- backend
uvicorn api_llm:app --reload --port 8000

# Terminal 2 -- frontend
streamlit run frontend_app.py
```

Abre `http://localhost:8501`. El backend carga BLIP (descripción) +
Qwen2.5-1.5B-Instruct (traducción y normalización al español); el
diagnóstico sana/leucémica se determina de forma determinística según el
tipo de célula mencionado en el caption (no depende del LLM).

Si tu GPU va justa de memoria con ambos modelos cargados, cambia
`LLM_MODEL_NAME` en `api_llm.py` a `"Qwen/Qwen2.5-0.5B-Instruct"`.

---

## Diagnóstico de sobreajuste (si el BLEU sale muy alto)

Si sospechas que el modelo memorizó el patrón rígido de las plantillas en vez
de generalizar, entrena la variante con menos capacidad entrenable (solo
LoRA, sin las capas completas) y compara:

```powershell
python train_hemblip_lora_only.py `
    --healthy_csv .\data_ft_train\healthy_captions.csv `
    --leukemic_csv .\data\leukemic_captions_faithful_train.csv `
    --output_dir .\checkpoints\hemblip_lora_only
```

---

## Estructura de carpetas esperada

```
TT/
├── bloodcells_dataset/              <- paso 1.2
│   ├── basophil/ ... neutrophil/
├── LeukemiaAttri/                   <- paso 1.3
│   └── LeukemiaAttri_Dataset/H_100X_C2/
│       ├── Images/{train,test}/
│       └── json_labels/{train,test}.json
├── pbc_attr_v1_{train,val,test}.csv <- paso 1.1
├── PKG-AML-Cytomorphology_LMU/       <- paso 9.1 (opcional)
├── data_ft_train/                   <- se genera (paso 2.1)
├── data/                            <- se genera (paso 2.2)
├── checkpoints/                     <- se genera al entrenar
├── eda_full_report/                 <- se genera (paso 2.3)
└── (todos los .py de este repo)
```

---

## Archivos vigentes vs. históricos

Los siguientes scripts se conservan en el repo por historial, pero **ya no
son parte del pipeline activo** (fueron reemplazados durante el desarrollo):
`train_hemblip_base.py`, `train_hemblip_base_single_stage.py`,
`build_leukemic_captions_coco.py`, `app.py`, `api.py`, `fix_and_export_model.py`.
Usa siempre los scripts listados en las secciones de arriba.

---

## Problemas comunes

| Síntoma | Causa / solución |
|---|---|
| `torch.cuda.is_available()` da `False` | Reinstala torch con el índice CUDA correcto (sección 0) |
| Warning sobre `tied_weights` al cargar el modelo | Inofensivo, no afecta la generación |
| `ValueError` al cargar el tokenizer de BioBERT (BERTScore) | `pip install sentencepiece protobuf` |
| Todo se clasifica como "sana" | Revisa que `BLAST_KEYWORDS` en `api_llm.py` siga alineado con los tipos de célula que usa tu dataset de entrenamiento |
| El servidor tarda horas en vez de minutos | Verifica que esté usando `cuda`, no `cpu` (aparece en la primera línea de log de cada script) |
