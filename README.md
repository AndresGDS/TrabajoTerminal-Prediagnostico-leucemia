---
title: HemVLM
emoji: 🩸
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# HemVLM -- Prediagnostico de leucemia

Prototipo academico que genera
descripciones morfologicas en lenguaje natural de celulas de sangre
periferica (sanas o leucemicas) a partir de una imagen de microscopio,
usando una arquitectura BLIP con LoRA en el decoder.

No reemplaza el diagnostico de un hematologo -- es una herramienta de
apoyo a la generacion preliminar de descripciones celulares.

---

## Datos y dependencias externas (no incluidos en este repo)

Este repositorio trae solo el codigo. Los datasets y el modelo base pesan
varios GB y hay que descargarlos aparte, en las rutas exactas que esperan
los scripts. Sigue estos pasos en orden.

### 1. Celulas sanas -- atributos (WBCAtt)

- **Fuente:** https://github.com/apple2373/wbcatt
- **Que descargar:** dentro de la carpeta `submission/` del repo, los archivos
  `pbc_attr_v1_train.csv`, `pbc_attr_v1_val.csv`, `pbc_attr_v1_test.csv`.
- **Donde colocarlo:** en la raiz del proyecto (junto a `build_captions.py`).

### 2. Celulas sanas -- imagenes (dataset PBC)

- **Fuente original:** Mendeley Data -- https://data.mendeley.com/datasets/snkd93bnjr/1
  ("A dataset for microscopic peripheral blood cell images for development
  of automatic recognition systems")
- **Alternativa mas facil de descargar:** Kaggle -- busca
  `unclesamulus/blood-cells-image-dataset` (mismo dataset, ya organizado en
  8 carpetas por tipo de celula: `basophil`, `eosinophil`, `erythroblast`,
  `ig`, `lymphocyte`, `monocyte`, `neutrophil`, `platelet`).
- **Donde colocarlo:** carpeta `bloodcells_dataset/` en la raiz del proyecto,
  con esas 8 subcarpetas dentro (el script `build_captions.py` las busca ahi).

### 3. Celulas leucemicas (LeukemiaAttri)

- **Repositorio del dataset:** https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024
- **Descarga real (Google Drive):** el link esta dentro del README de ese
  repo -- llega dividido en varios .zip por el limite de tamano de Drive;
  extrae todos en el MISMO destino para que se reconstruya el arbol completo.
- **Subconjunto a usar:** `H_100X_C2` (camara HD1500T, 100x -- es el unico
  anotado directamente por hematologos; los demas subconjuntos tienen
  atributos "transferidos" automaticamente y son menos confiables).
- **Donde colocarlo:** `LeukemiaAttri/LeukemiaAttri_Dataset/H_100X_C2/`,
  con las subcarpetas `Images/train`, `Images/test` y `json_labels/`
  (`train.json`, `test.json`) tal como vienen en el zip.

### 4. PyTorch con soporte CUDA (si vas a entrenar/inferir con GPU)

El `pip install torch` normal instala una build CPU-only en algunos casos.
Para GPU NVIDIA, instala explicitamente la build con CUDA que corresponda
a tu version de Python y de driver:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Si da error de "no matching distribution", prueba con `cu128` en vez de
`cu124`. Verifica que quedo activo con:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. Modelo base BLIP (se descarga solo)

No requiere descarga manual: la primera vez que corras
`train_hemblip_base_single_stage.py` (o cualquier script que llame a
`build_model()`), `transformers` descarga automaticamente
`Salesforce/blip-image-captioning-base` (~1GB) desde Hugging Face y lo deja
en cache local. Solo necesitas conexion a internet esa primera vez.

### 6. Resto de dependencias de Python

```powershell
pip install -r requirements.txt
```

---

## Estructura de carpetas esperada (resumen)

```
TT/
├── bloodcells_dataset/          <- paso 2
│   ├── basophil/ ... platelet/
├── LeukemiaAttri/
│   └── LeukemiaAttri_Dataset/
│       └── H_100X_C2/           <- paso 3
│           ├── Images/{train,test}/
│           └── json_labels/{train,test}.json
├── pbc_attr_v1_train.csv        <- paso 1
├── pbc_attr_v1_val.csv
├── pbc_attr_v1_test.csv
├── data/                        <- se genera con build_captions.py y
│                                    build_leukemic_captions_coco.py
├── checkpoints/                 <- se genera al entrenar
└── (todos los .py de este repo)
```

