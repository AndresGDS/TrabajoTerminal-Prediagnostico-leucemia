# Documentación técnica — Sistema HemVLM (línea base)

**Trabajo Terminal 2027-A140 — Modelo de lenguaje y visión para el prediagnóstico de leucemia**
Estado: réplica de la arquitectura base de HemBLIP completa. Pendiente: sustitución del encoder visual (ViT-B/16 → FastViT).

---

## 1. Resumen general

El sistema genera descripciones morfológicas en lenguaje natural a partir de imágenes de células sanguíneas (sanas o leucémicas), replicando la arquitectura del paper *HemBLIP: A Vision–Language Model for Interpretable Leukemia Cell Morphology Analysis* (van Logtestijn & Manescu, 2026) como línea de desempeño inicial, antes de sustituir su encoder visual.

El pipeline completo tiene cinco etapas: **(1)** datos crudos de dos fuentes distintas → **(2)** construcción de captions morfológicos → **(3)** arquitectura BLIP con LoRA → **(4)** entrenamiento → **(5)** modelo final listo para inferencia.

---

## 2. Datasets utilizados

### 2.1 WBCAtt (células sanas)

- **Fuente:** repositorio público `apple2373/wbcatt` (NeurIPS 2023), basado en el dataset PBC (Barcelona).
- **Estructura original:** CSV (`pbc_attr_v1_train.csv`, `_val.csv`, `_test.csv`) con una fila por célula ya recortada, 11 atributos categóricos en texto (`cell_size`, `cell_shape`, `nucleus_shape`, `nuclear_cytoplasmic_ratio`, `chromatin_density`, `cytoplasm_vacuole`, `cytoplasm_texture`, `cytoplasm_colour`, `granule_type`, `granule_colour`, `granularity`) más `label` (tipo de célula) y `path` (ruta relativa con subcarpeta por tipo).
- **Imágenes:** carpeta `bloodcells_dataset/<tipo_celula>/`, todas en **360×363 px**, sin valores nulos.
- **Filas usadas:** 6,169 (split `train`).
- **Distribución por tipo:** Neutrophil 1,984 · Eosinophil 1,876 · Monocyte 829 · Basophil 744 · Lymphocyte 736.
- **¿Se modificó el dataset?** No se alteraron las imágenes originales. Se generó un archivo derivado (`healthy_captions.csv`) que traduce cada fila de atributos a una oración en inglés (ver §3.1). No se aplicó resize, aumento de datos ni filtrado adicional en esta fase.

### 2.2 LeukemiaAttri (células leucémicas)

- **Fuente:** repositorio `intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024`, descargado desde Google Drive (llegó dividido en 5 zips por límite de tamaño de Drive, sin relación con subconjuntos del dataset).
- **Estructura original:** el dataset completo tiene 12 subconjuntos (`H_100X_C1/C2`, `H_10X_C1/C2`, `H_40X_C1/C2`, `L_100X_C1/C2`, `L_10X_C1/C2`, `L_40X_C1/C2`) según microscopio (H=caro, L=barato), resolución (10x/40x/100x) y cámara (C1/C2). Se usó **`H_100X_C2`**, el subconjunto con anotación directa de hematólogo (los demás tienen atributos "transferidos" automáticamente y son menos confiables).
- **Formato:** COCO JSON (`json_labels/train.json`, `test.json`) — imágenes de campo completo de **640×640 px**, cada una con varias anotaciones (una por célula), cada anotación con su `bbox`, `category_name` (tipo de célula) y 7 atributos numéricos (`cell_size`, `nuclear_chromatio`, `nuclear_shape`, `nucleolus`, `cytoplasm`, `cytoplasmic_basophilia`, `cytoplasmic_vacuoles`).
- **Diagnóstico:** no viene como campo explícito; se extrae del nombre de archivo (sufijo `_ALL`, `_AML`, `_CML`, `_CLL`, `_APML`).
- **¿Se modificó el dataset?** Sí, de forma necesaria para adaptarlo al formato célula-por-imagen que requiere BLIP:
  1. **Recorte (crop):** cada célula se extrajo de su imagen de campo completo usando el `bbox` de su anotación, generando una imagen nueva por célula (`data/leukemic_crops/`).
  2. **Pérdida de datos conocida:** uno de los 5 zips de Drive (el n.º 3) no se extrajo correctamente, causando que ~19–25 % de las anotaciones no encontraran su imagen fuente. Esas anotaciones se omitieron automáticamente (no se interrumpió el proceso). Total final: **8,200 recortes** (6,559 de train + 1,641 de test).
  3. **Valor `4` en atributos = "no aplica":** el EDA reveló que el código `4` aparece exactamente 2,199 veces en cada atributo, coincidiendo con la categoría `category_name = "none"` — es decir, corresponde a detecciones que no son una célula real (artefactos), no a un nivel morfológico. Esto quedó documentado pero **no se filtró aún** de los captions generados.
  4. **Legend de atributos numéricos:** los valores `0`, `1`, `2` de cada atributo (aparte del `4`) no tienen un diccionario oficial de significado disponible en el paper principal (remite a material suplementario no accesible). Se usó un diccionario **placeholder** (`ATTR_VALUE_LABELS` en `build_leukemic_captions_coco.py`) con valores razonables pero no confirmados; los valores fuera del placeholder se muestran como `"level N"`. **Esto es una limitación pendiente de resolver**, documentable como tal en el reporte.
- **Distribución de diagnóstico:** AML 767 · ALL 708 · CML 110 · CLL 48 · APML 44 (fuerte desbalance hacia AML/ALL, esperado clínicamente).
- **Balance sano vs. leucémico final:** 6,169 (43%) vs. 8,200 (57%) — razonablemente equilibrado.

### 2.3 Calidad de imagen (EDA exhaustivo)

Se construyó `eda_full_dataset.py` para revisar, célula por célula (no una muestra), modo de color (RGB vs. escala de grises), corrupción, duplicados exactos (hash MD5), consistencia de resolución y metadatos inusuales (perfil ICC, EXIF, transparencia). Los resultados de esa corrida se documentan en `eda_full_report/eda_summary.txt` una vez ejecutada — **pendiente de incorporar aquí los valores finales** si aún no se ha corrido o revisado.

---

## 3. Arquitectura del modelo

**Modelo base:** `Salesforce/blip-image-captioning-base` (arquitectura BLIP, Li et al. 2022), el mismo tipo de arquitectura que usa HemBLIP en el paper original antes de cualquier sustitución de encoder.

- **Encoder visual:** ViT-B/16, resolución fija 224×224. **Completamente congelado** (`requires_grad=False` en todos sus parámetros) — nunca se actualiza durante el entrenamiento, tal como especifica el método replicado.
- **Decoder de texto:** Transformer tipo BERT (BERT-base) conectado al encoder visual mediante *cross-attention* — en cada paso de generación de texto, el decoder "consulta" las características visuales extraídas por el ViT.
- **Adaptación LoRA:** se aplica **solo sobre el decoder**, en las proyecciones de atención `query`, `key`, `value` y `output.dense` (tanto self-attention como cross-attention), con rank=16, alpha=32, dropout=0.05. Esto entrena una fracción muy pequeña de parámetros en vez del modelo completo, igual que en el paper.
- **Optimizador:** AdamW, learning rate 5e-5, con early stopping basado en la pérdida de validación (patience configurable, por defecto 3 épocas sin mejora).

### 3.1 Cómo se generan los captions de entrenamiento

Los captions **no se generan con un LLM ni se copian del paper** — se construyen de forma templada (reglas fijas) a partir de los atributos categóricos de cada dataset:

- **Sanas:** `"This is a {tipo}. It shows a {tamaño} cell size, {forma} overall shape, a {forma_núcleo} nucleus, ..."` — una cláusula por atributo presente.
- **Leucémicas:** `"This cell shows features consistent with {diagnóstico}. Morphological features: {atributo1}, {atributo2}, ..."`.

Esto es una simplificación del método del paper (que usa plantillas + paráfrasis con GPT-4); aquí solo se usa la parte de plantillas, sin paráfrasis adicional.

---

## 4. Archivos del proyecto y su función

| Archivo | Función |
|---|---|
| `build_captions.py` | Lee `pbc_attr_v1_*.csv` (WBCAtt) y genera `healthy_captions.csv` (`image_path,caption`), resolviendo la subcarpeta correcta por tipo de célula. |
| `build_leukemic_captions_coco.py` | Lee el JSON COCO de LeukemiaAttri, **recorta cada célula por su bbox**, guarda el recorte y genera `leukemic_captions.csv`. |
| `eda_dataset.py` | EDA por muestreo: distribución de clases y atributos categóricos de ambos datasets, balance binario, resolución de imagen (muestra de 50). |
| `eda_full_dataset.py` | EDA exhaustivo sobre el 100% de las imágenes usadas para entrenar: corrupción, modo de color, duplicados (MD5), tamaño de archivo, metadatos. Guarda CSV detallado + `eda_summary.txt`. |
| `train_hemblip_base.py` | Entrena el modelo en **dos etapas secuenciales** (sanas → leucémicas), cada una con su propio early stopping. Incluye el fix de fusión LoRA antes de guardar. |
| `train_hemblip_base_single_stage.py` | Entrena en **un solo paso** sobre ambos datasets mezclados y barajados — fiel al método original del paper (sin currículum). |
| `fix_and_export_model.py` | Repara un modelo ya entrenado cuyo guardado final quedó corrupto por el bug de PEFT (ver §5.2): reconstruye la arquitectura, carga el checkpoint `.pt`, funde LoRA, y re-exporta correctamente. No requiere reentrenar. |
| `generate_caption.py` | Carga un modelo ya entrenado y genera la descripción de una imagen suelta (prueba rápida / inferencia individual). |
| `requirements.txt` | Dependencias: torch (build CUDA), transformers, peft, pandas, pillow, scikit-learn, tqdm, matplotlib. |

---

## 5. Estrategias de entrenamiento y hallazgos

### 5.1 Dos etapas vs. un solo paso

Se entrenaron **dos variantes** para comparar contra el método real del paper:

1. **Dos etapas** (recomendación recibida externamente): primero se entrena hasta convergencia solo con sanas, luego se retoman esos pesos y se continúa entrenando solo con leucémicas.
2. **Un solo paso** (método del paper): se combinan y barajan ambos datasets desde el inicio, entrenando una sola vez sobre el conjunto completo.

**Hallazgo:** al probar el modelo de dos etapas con `generate_caption.py`, se observó **olvido catastrófico** — sobre una imagen sana (neutrófilo), el modelo solo devolvía *"this is a neutrophil"* (sin ningún atributo morfológico, perdiendo el estilo detallado de la etapa 1), mientras que sobre una imagen leucémica sí generaba el caption completo con atributos (estilo de la etapa 2, entrenada al final). Esto sugiere que el currículum secuencial sin mecanismo de "repaso" hace que el modelo sobrescriba lo aprendido en la primera etapa. Es un hallazgo documentable para la sección de análisis cualitativo de errores del reporte técnico.

### 5.2 Bug corregido: guardado del modelo con LoRA

Al aplicar LoRA sobre el decoder, PEFT renombra internamente los pesos (agrega el prefijo `base_model.model...`). El primer intento de guardar el modelo final con `save_pretrained()` guardó esos nombres tal cual, lo que impedía recargar el modelo correctamente como un `BlipForConditionalGeneration` estándar — el decoder terminaba con pesos aleatorios (el caption generado era texto sin sentido).

**Corrección:** antes de guardar, se fusionan los pesos LoRA de vuelta al decoder original con `merge_and_unload()`, quedando en formato estándar. Este fix ya está incorporado en ambos scripts de entrenamiento; para el modelo ya entrenado antes del fix, se usó `fix_and_export_model.py` para repararlo sin reentrenar.

---

## 6. Estado actual

- Réplica de la arquitectura base de HemBLIP (BLIP + LoRA en decoder, ViT-B/16 congelado) completa y funcional de punta a punta (imagen → texto).
- Dos variantes de entrenamiento comparables (2 etapas vs. 1 paso).
- Pipeline de datos reproducible para ambos datasets (sanas y leucémicas).
- Pendiente: legend oficial de valores numéricos de atributos de LeukemiaAttri.
- Pendiente: evaluación cuantitativa formal (BLEU, ROUGE-L, BERTScore) sobre un set de prueba, como hace el paper original.
- Pendiente: filtrar de los captions las anotaciones marcadas como "no aplica" (valor `4`) antes de un reentrenamiento futuro.
- Siguiente objetivo de la propuesta de TT: sustituir el encoder visual ViT-B/16 por FastViT y comparar el desempeño descriptivo y diagnóstico contra esta línea base.

---

## 7. Limitaciones conocidas (para la sección de discusión del reporte)

1. Los captions son generados por plantillas fijas, no por un LLM parafraseando (a diferencia del método completo del paper, que usa GPT-4 para variar la redacción).
2. ~19–25% de las anotaciones de LeukemiaAttri se perdieron por un problema de extracción de zip, no por limitación del dataset en sí.
3. El significado exacto de los códigos numéricos de atributos de LeukemiaAttri no está confirmado (se usó un placeholder razonable).
4. El entrenamiento en dos etapas presenta olvido catastrófico observable cualitativamente; no se ha cuantificado con métricas formales todavía.
5. No se ha corrido evaluación cuantitativa (BLEU/ROUGE-L/BERTScore) sobre ningún set de prueba separado — las pruebas hechas hasta ahora son cualitativas, sobre ejemplos individuales.
