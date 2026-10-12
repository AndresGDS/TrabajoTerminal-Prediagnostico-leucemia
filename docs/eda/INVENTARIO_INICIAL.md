> Copia para repositorio: rutas personales sustituidas por `<TT>` y captions de muestra omitidos. Se conservan historia y correcciones; original íntegro local.

# Inventario inicial de HemVLM

Fecha: 2026-10-08. Alcance: exclusivamente el arbol local TT.

Raiz absoluta: `<TT>`. Todas las rutas siguientes son relativas a esta raiz.

## Metodo y limites

Conteos nuevos mediante recorrido completo de archivos, lectura completa de filas CSV (sin encabezado) y listado de entradas no-directorio de cada ZIP. Las cantidades de imagenes cuentan archivos por extension, no imagenes unicas ni celulas. No se verificaron integridad visual, hashes, duplicados, cobertura de anotaciones ni fuga entre particiones. Los contenidos de JSON/TXT se describen mediante muestras. Se leyeron JSON en memoria para seleccionar sus primeros registros; no se calcularon estadisticas de anotaciones. No se ejecutaron notebooks ni scripts existentes, ni se extrajeron ZIP, limpiaron datos, generaron recortes/captions o cambiaron particiones. No se tomaron cifras de informes anteriores.

Se encontraron 66238 archivos fisicos antes de crear este inventario. Este total incluye copias y contenedores y NO representa el tamano de un dataset utilizable.

## Rutas principales

| Componente | Ruta desde TT | Contenido localizado |
|---|---|---|
| WBCAtt | `Dataset/WBCAtt/` | Tres CSV train, val y test |
| PBC | `bloodcells_dataset/` | JPG en ocho carpetas de tipos celulares |
| LeukemiaAttri principal | `Dataset/LeukemiaAttri/` | Cinco carpetas numeradas 001 a 005; PNG, JSON y ZIP de etiquetas |
| Otra carpeta LeukemiaAttri | `Dataset Andres/Dataset Andres/LeukemiaAttri/` | Estructura parcialmente extraida, con cantidades diferentes |
| Otra carpeta de imagenes | `LeukemiaAttri_Dataset/` | PNG en carpetas de entrenamiento, sin JSON/TXT sueltos |
| Captions candidatos de Arturo | `Dataset/captions/captions.csv` | Procedencia pendiente de confirmar; tambien existe `captions.zip` |
| Protocolo | `protocolo/REESTRUC_140 (2).pdf` | Localizado; contenido no analizado |

## Carpetas extraidas: conteos completos

| Carpeta | Archivos por extension |
|---|---|
| `Dataset/LeukemiaAttri` | 20 .json, 25790 .png, 12 .zip |
| `Dataset/WBCAtt` | 3 .csv |
| `Dataset/captions` | 1 .csv |
| `Dataset Andres` | 20 .json, 20031 .png, 13 .zip |
| `LeukemiaAttri_Dataset` | 3246 .png |
| `bloodcells_dataset` | 17092 .jpg |

### Fragmentos de LeukemiaAttri

En la carpeta principal, cada fragmento contiene `LeukemiaAttri_Dataset/<subconjunto>/<Images o images>/<train o test>`. Se conservan tambien variantes `Train` y `Test`. No se han combinado los fragmentos. Los JSON sueltos estan en el fragmento 001, bajo `<subconjunto>/json_labels/{train,test}.json`.

| Ruta del fragmento | PNG | JSON | ZIP |
|---|---:|---:|---:|
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001` | 5759 | 20 | 12 |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-002` | 5601 | 0 | 0 |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-003` | 5693 | 0 | 0 |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-004` | 5470 | 0 | 0 |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-005` | 3267 | 0 | 0 |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001` | 0 | 20 | 13 |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-002` | 5601 | 0 | 0 |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-003` | 5693 | 0 | 0 |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-004` | 5470 | 0 | 0 |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-005` | 3267 | 0 | 0 |

### Imagenes principales por subconjunto y particion fisica

Suma de archivos en los cinco fragmentos; no deduplicada. Las mayusculas se agrupan solo para mostrar el conteo, sin renombrar archivos.

| Subconjunto | train/Train | test/Test |
|---|---:|---:|
| H_100X_C1 | 1676 | 692 |
| H_100X_C2 | 1677 | 692 |
| H_10X_C1 | 1608 | 657 |
| H_10X_C2 | 1677 | 692 |
| H_40X_C1 | 1639 | 676 |
| H_40X_C2 | 1667 | 685 |
| L_100X_C1 | 1674 | 691 |
| L_100X_C2 | 1638 | 648 |
| L_10X_C2 | 1732 | 639 |
| L_40X_C1 | 1677 | 688 |
| L_40X_C2 | 1675 | 690 |

`L_10X_C1` aparece como carpeta vacia en el fragmento 001. No se localizaron PNG con ese subconjunto en el recorrido. Esto no demuestra que deba existir material para esa combinacion.

### PBC

| Carpeta | JPG |
|---|---:|
| `basophil` | 1218 |
| `eosinophil` | 3117 |
| `erythroblast` | 1551 |
| `ig` | 2895 |
| `lymphocyte` | 1214 |
| `monocyte` | 1420 |
| `neutrophil` | 3329 |
| `platelet` | 2348 |

Total: **17,092 JPG**. No se abrieron ni validaron todas las imagenes.

## CSV: conteos completos y muestras

### `Dataset/captions/captions.csv`

Conteo completo: **68,652 filas de datos**. Columnas (11): `filename`, `image_path`, `dataset`, `split`, `cell_type`, `diagnosis`, `magnification`, `diagnosis_idx`, `cell_type_idx`, `caption`, `caption_with_image_token`.

Primera fila (muestra; no resumen estadistico):

```json
{
  "filename": "BA_100102.jpg",
  "image_path": "data\\PBC_dataset_normal_DIB\\PBC_dataset_normal_DIB\\basophil\\BA_100102.jpg",
  "dataset": "WBCATt",
  "split": "",
  "cell_type": "basophil",
  "diagnosis": "",
  "magnification": "",
  "diagnosis_idx": "-1",
  "cell_type_idx": "0",
  "caption": "[Texto completo omitido en la copia del repositorio; permiso pendiente]",
  "caption_with_image_token": "[Texto completo omitido en la copia del repositorio; permiso pendiente]"
}
```

### `Dataset/WBCAtt/pbc_attr_v1_test.csv`

Conteo completo: **3,099 filas de datos**. Columnas (14): `img_name`, `label`, `cell_size`, `cell_shape`, `nucleus_shape`, `nuclear_cytoplasmic_ratio`, `chromatin_density`, `cytoplasm_vacuole`, `cytoplasm_texture`, `cytoplasm_colour`, `granule_type`, `granule_colour`, `granularity`, `path`.

Primera fila (muestra; no resumen estadistico):

```json
{
  "img_name": "EO_501290.jpg",
  "label": "Eosinophil",
  "cell_size": "small",
  "cell_shape": "round",
  "nucleus_shape": "unsegmented-band",
  "nuclear_cytoplasmic_ratio": "low",
  "chromatin_density": "densely",
  "cytoplasm_vacuole": "no",
  "cytoplasm_texture": "clear",
  "cytoplasm_colour": "light blue",
  "granule_type": "round",
  "granule_colour": "red",
  "granularity": "yes",
  "path": "PBC_dataset_normal_DIB/eosinophil/EO_501290.jpg"
}
```

### `Dataset/WBCAtt/pbc_attr_v1_train.csv`

Conteo completo: **6,169 filas de datos**. Columnas (14): `img_name`, `label`, `cell_size`, `cell_shape`, `nucleus_shape`, `nuclear_cytoplasmic_ratio`, `chromatin_density`, `cytoplasm_vacuole`, `cytoplasm_texture`, `cytoplasm_colour`, `granule_type`, `granule_colour`, `granularity`, `path`.

Primera fila (muestra; no resumen estadistico):

```json
{
  "img_name": "BNE_7323.jpg",
  "label": "Neutrophil",
  "cell_size": "big",
  "cell_shape": "round",
  "nucleus_shape": "unsegmented-band",
  "nuclear_cytoplasmic_ratio": "low",
  "chromatin_density": "densely",
  "cytoplasm_vacuole": "no",
  "cytoplasm_texture": "clear",
  "cytoplasm_colour": "light blue",
  "granule_type": "small",
  "granule_colour": "pink",
  "granularity": "yes",
  "path": "PBC_dataset_normal_DIB/neutrophil/BNE_7323.jpg"
}
```

### `Dataset/WBCAtt/pbc_attr_v1_val.csv`

Conteo completo: **1,030 filas de datos**. Columnas (14): `img_name`, `label`, `cell_size`, `cell_shape`, `nucleus_shape`, `nuclear_cytoplasmic_ratio`, `chromatin_density`, `cytoplasm_vacuole`, `cytoplasm_texture`, `cytoplasm_colour`, `granule_type`, `granule_colour`, `granularity`, `path`.

Primera fila (muestra; no resumen estadistico):

```json
{
  "img_name": "BNE_422954.jpg",
  "label": "Neutrophil",
  "cell_size": "small",
  "cell_shape": "round",
  "nucleus_shape": "unsegmented-band",
  "nuclear_cytoplasmic_ratio": "low",
  "chromatin_density": "densely",
  "cytoplasm_vacuole": "no",
  "cytoplasm_texture": "clear",
  "cytoplasm_colour": "light blue",
  "granule_type": "small",
  "granule_colour": "pink",
  "granularity": "yes",
  "path": "PBC_dataset_normal_DIB/neutrophil/BNE_422954.jpg"
}
```

WBCAtt suma **10,298 filas**: train 6,169; val 1,030; test 3,099. Sus 14 columnas contienen nombre, etiqueta celular, 11 atributos morfologicos y ruta. La particion se identifica por el CSV. En la primera fila de cada CSV se comprobo que el archivo existe sustituyendo `PBC_dataset_normal_DIB/` por `bloodcells_dataset/`; son solo tres comprobaciones, no una validacion de todas las filas.

Captions contiene **68,652 filas**, no necesariamente ese numero de imagenes distintas. En las dos primeras filas se repite BA_100102.jpg con textos diferentes; `split`, `diagnosis` y `magnification` estan vacios. `image_path` utiliza el prefijo `data\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB\`, diferente de la ubicacion local. `caption_with_image_token` agrega `<image>` al texto en esas muestras. La procedencia de Arturo y la relacion completa con las anotaciones originales siguen pendientes; no se compararon semanticamente.

## JSON: organizacion observada en muestras

Se inspeccionaron los primeros registros de `H_100X_C2/json_labels/test.json` y `L_40X_C1/json_labels/test.json`, ambos bajo `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/`.

Ambos tienen estructura tipo COCO: `info`, `licenses`, `images`, `annotations`, `categories`. `images` identifica archivos con `id`, `file_name`, `width`, `height`; `annotations` enlaza por `image_id` e incluye `category_id`, `bbox`, `area`, `iscrowd`, `cell_id`, `category_name` y atributos numericos. `categories` relaciona identificadores y nombres celulares. Una imagen puede tener varias anotaciones; no se deben confundir imagenes con celulas.

Ejemplo de H_100X_C2: primera imagen `28_1_1000_CML.png`, 640 x 640 segun JSON. Primera anotacion: `image_id=0`, `category_id=8`, `category_name=myelocyte`, `bbox=[114,373,78,150]`. Atributos: `cell_size`, `nuclear_chromatio` (escritura original), `nuclear_shape`, `nucleolus`, `cytoplasm`, `cytoplasmic_basophilia`, `cytoplasmic_vacuoles`. No se ha validado el significado de cada codigo ni se generaliza el tamano de imagen de esta muestra.

## Comprimidos: inventario separado

Cada fila cuenta entradas de archivo del ZIP sin extraerlo. Un ZIP interno cuenta como una entrada; no se suma recursivamente su contenido al contenedor. Estos numeros NO se agregan a las carpetas extraidas como observaciones nuevas. Igualdad de cantidades no demuestra igualdad de contenido.

| ZIP (ruta desde TT) | Entradas de archivo | Formatos internos |
|---|---:|---|
| `captions.zip` | 1 | 1 .csv |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_100X_C1/txt_labels.zip` | 4766 | 4766 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_100X_C2/txt_labels.zip` | 4748 | 4748 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_10X_C1/txt_labels.zip` | 4530 | 4530 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_10X_C2/txt_labels.zip` | 4738 | 4738 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_40X_C1/txt_labels.zip` | 4630 | 4630 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_40X_C2/txt_labels.zip` | 4700 | 2 .json, 4698 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_100X_C1/txt_labels.zip` | 4732 | 2 .json, 4730 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_100X_C2/labels.zip` | 4574 | 2 .json, 4572 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_10X_C2/txt_labels.zip` | 4742 | 4742 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C1/attridet.zip` | 2349 | 2349 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C1/WBC_detection.zip` | 2349 | 2349 .txt |
| `Dataset/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C2/text_label.zip` | 4730 | 4730 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_100X_C1/txt_labels.zip` | 4766 | 4766 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_100X_C2/txt_labels.zip` | 4748 | 4748 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_10X_C1/txt_labels.zip` | 4530 | 4530 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_10X_C2/txt_labels.zip` | 4738 | 4738 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_40X_C1/txt_labels.zip` | 4630 | 4630 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/H_40X_C2/txt_labels.zip` | 4700 | 2 .json, 4698 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_100X_C1/txt_labels.zip` | 4732 | 2 .json, 4730 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_100X_C2/labels.zip` | 4574 | 2 .json, 4572 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_10X_C2/txt_labels.zip` | 4742 | 4742 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C1/attridet.zip` | 2349 | 2349 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C1/WBC_detection.zip` | 2349 | 2349 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset/L_40X_C2/text_label.zip` | 4730 | 4730 .txt |
| `Dataset Andres/Dataset Andres/LeukemiaAttri/LeukemiaAttri_Dataset-20260907T175427Z-1-001/LeukemiaAttri_Dataset.zip` | 32 | 20 .json, 12 .zip |
| `Dataset Andres.zip` | 25825 | 3 .csv, 20 .json, 25790 .png, 12 .zip |
| `Dataset.zip` | 25826 | 3 .csv, 1 .ipynb, 20 .json, 25790 .png, 12 .zip |

`Dataset.zip` contiene ademas `Dataset/EDA_Leukemia_FastVLM_reestructurado.ipynb` y los tres CSV de WBCAtt. `Dataset Andres.zip` contiene los tres CSV, aunque estos no se localizaron sueltos bajo `Dataset Andres/`. El ZIP `LeukemiaAttri_Dataset.zip` ubicado dentro de esa carpeta contiene 20 JSON y 12 ZIP, sin PNG directos. `captions.zip` contiene una entrada `captions.csv`; no se ha comprobado igualdad de bytes con el CSV extraido.

### TXT: lectura dentro de ZIP, sin extraccion

No se localizaron TXT sueltos en TT. Se leyeron hasta dos lineas del primer TXT de cada uno de los 12 ZIP de etiquetas de la carpeta principal, fragmento 001.

- `H_100X_C2/txt_labels.zip`: `txt_labels/WBC_Detection/test/45_30_1000_AML.txt`, primera linea `3 0.612 0.459 0.092 0.188` (cinco campos).
- `H_10X_C2/txt_labels.zip`: `txt_labels/AttriDet/test/6_1_100_ALL.txt`, primera linea `12 0.445 0.323 0.111 0.188 1 1 1 0 1 1 0` (doce campos).
- En las muestras aparecen esquemas compatibles con deteccion tipo YOLO y con deteccion mas siete atributos. La semantica exacta, orden, normalizacion y correspondencia de clases con COCO requieren documentacion; no se asumen equivalentes sus identificadores.
- Algunas rutas denominadas AttriDet tambien muestran cinco campos en la muestra (`H_10X_C1`); el nombre de carpeta no basta para garantizar el esquema.
- `L_100X_C2/labels.zip` contiene `json_labels/train.json` y `json_labels/test.json`, aunque ese subconjunto no tiene JSON sueltos en la carpeta principal. Tambien hay JSON dentro de los ZIP de H_40X_C2 y L_100X_C1. No se han comparado con los JSON extraidos.

## Notebooks, scripts y documentos

Notebooks locales encontrados (tres archivos fisicos; identidad de contenido no comprobada):

- `EDA_Leukemia_FastVLM_.ipynb`.
- `EDA_Leukemia_FastVLM_reestructurado.ipynb`.
- `protocolo/EDA_Leukemia_FastVLM_reestructurado.ipynb`.

No se encontraron scripts sueltos (.py, .R, .sh, .ps1, entre otros) en el recorrido: los formatos encontrados fueron PNG, JPG, JSON, CSV, ZIP, IPYNB, PDF y MD. No se ejecutaron notebooks ni se usaron sus salidas como evidencia.

Documentos locales adicionales (solo localizados, no utilizados como resultados actuales):

- `protocolo/REESTRUC_140 (2).pdf`.
- `protocolo/avance_fase_II_ingenieria_datos_20261006/INFORME_AVANCE_FASE_II.md`.
- `protocolo/avance_fase_II_ingenieria_datos_20261006/MATRIZ_TRAZABILIDAD.md`.
- `protocolo/avance_fase_II_ingenieria_datos_20261006/PENDIENTES_CIERRE_FASE_II.md`.

Se consulto solamente la pagina del [repositorio de referencia](https://github.com/AndresGDS/TrabajoTerminal-Prediagnostico-leucemia). Su listado contiene scripts de construccion de captions, EDA y entrenamiento HemBLIP. Son recursos remotos, no scripts encontrados localmente. No se descargaron ni ejecutaron.

## Relacion de componentes y pendientes

WBCAtt describe atributos de imagenes PBC mediante nombres/rutas. LeukemiaAttri combina imagenes de campos con anotaciones de objetos y atributos; la imagen y la celula anotada no son necesariamente la misma unidad. Captions aporta textos y referencias a imagenes que deberan contrastarse despues con las fuentes. No se ha construido aun un dataset para HemBLIP.

1. Confirmar si `Dataset/captions/captions.csv` es el original recibido de Arturo. Si no, solicitar su ubicacion.
2. No se localizaron scripts locales: si se esperaban, solicitar su ubicacion dentro de TT; el repositorio remoto queda identificado como referencia.
3. Resolver posteriormente la relacion entre los cinco fragmentos, la carpeta Dataset Andres, la carpeta LeukemiaAttri_Dataset y los ZIP. No considerar ninguna copia completa o unica solo por su nombre; tampoco sumar copias como nuevas imagenes.
4. Verificar cobertura de anotaciones y de imagenes, identidad de copias y pertenencia de particiones antes de reconstruir. Revisar L_10X_C1 vacio sin concluir que faltan datos obligatorios.
5. Documentar codigos de atributos y clases TXT/JSON, variantes de nombres y rutas, y correspondencia entre unidades imagen/celula/caption.
6. Revisar el protocolo y acordar el alcance cientifico despues del inventario. No se seleccionaron ejemplos, ni se buscaron datasets adicionales, ni se hizo el EDA completo.

Los componentes de datos solicitados tienen ubicaciones candidatas localizadas. Las dudas de procedencia y disponibilidad de scripts quedan explicitamente pendientes.


## Nota de corrección y procedencia — 2026-10-08

Se conserva íntegro el contenido anterior como registro histórico. Esta nota corrige o actualiza expresamente estas afirmaciones del inventario:

1. **Corrección de la muestra de captions:** la afirmación «En las dos primeras filas se repite BA_100102.jpg con textos diferentes» es incorrecta. Las primeras dos filas de datos corresponden a **BA_100102.jpg** y **BA_101381.jpg**, respectivamente; son nombres distintos. No debe usarse aquella afirmación como evidencia de duplicación.
2. **Procedencia confirmada por el usuario:** el usuario confirmó que `Dataset/captions/captions.csv` fue extraído del ZIP de captions proporcionado por Arturo. Las menciones previas a una procedencia pendiente quedan actualizadas por esta confirmación. Es un **testimonio/confirmación del usuario, no una verificación independiente de contenido, autoría ni igualdad byte a byte entre CSV y ZIP**.

No se cambian retrospectivamente los conteos del inventario ni se sustituyen sus observaciones por resultados del EDA.
