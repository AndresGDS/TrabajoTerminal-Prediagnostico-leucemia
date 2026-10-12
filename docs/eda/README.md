# EDA de WBCAtt/PBC y LeukemiaAttri para HemVLM

Preparación local de la rama `karina/eda-wbcatt-leukemiaattri`, sin commit ni push.
Base: `main`, commit `f46fea8225dd0e85e21b111ad9d7892f314ac777`.
Remoto: https://github.com/AndresGDS/TrabajoTerminal-Prediagnostico-leucemia.git

## Objetivo y alcance

Comprender fuentes, unidades, cobertura, calidad, duplicados y captions antes de
preparar pares imagen–texto. Este aporte no reconstruye el dataset final, no limpia
fuentes, no entrena modelos y no valida clínicamente etiquetas. La búsqueda de
datasets adicionales no forma parte de estos EDA.

No se encontró una copia Git dentro de TT; se clonó el repositorio en una carpeta
hermana de Dataset, inicialmente limpia. No se encontraron AGENTS.md ni instrucciones
de contribución adicionales. Se revisaron README.md, requirements.txt y .gitignore.
Las instrucciones del README principal describen otro flujo de preparación y
entrenamiento; para estos EDA se utiliza la configuración siguiente, conservando
los cinco fragmentos separados y los once subconjuntos disponibles.

## Archivos propuestos

| Ubicación | Contenido |
|---|---|
| `../../notebooks/eda/01_EDA_WBCAtt.ipynb` | Versión vigente de WBCAtt; reglas semánticas actuales |
| `../../notebooks/eda/02_EDA_LeukemiaAttri.ipynb` | Versión vigente, incluida sección 12 de desgloses |
| `../../notebooks/eda/eda_paths.py` | Único módulo auxiliar: configuración de rutas, sin leer datos al importar |
| `../../requirements-eda.txt` | Versiones registradas del entorno de los EDA, separadas de entrenamiento |
| `INVENTARIO_INICIAL.md` | Inventario histórico y correcciones; copia sin ruta personal ni caption íntegro de muestra |
| `INFORME_EDA_WBCAtt_LeukemiaAttri.pdf` | Informe de 18 páginas, copia para repositorio con omisiones señaladas |
| `INFORME_EDA_WBCAtt_LeukemiaAttri.docx` | Fuente editable equivalente; sin microfotografías ni captions íntegros |
| `figuras/wbcatt_*.png` | Tres gráficas agregadas existentes, sin fotografías celulares |
| `figuras/leukemiaattri_*.png` | Tres gráficas agregadas existentes, sin fotografías celulares |
| `REDISTRIBUCION.md` | Fuentes consultadas, atribución y permisos pendientes |
| `MANIFIESTO_EDA.json` | Tamaños, hashes de fuentes/copias, celdas adaptadas y verificación |

El manifiesto registra tamaños exactos en bytes y SHA-256. Una huella verifica la
conservación de un archivo, no su calidad científica ni los derechos de uso.

## Procedencia de resultados guardados

- WBCAtt: ejecución completa local `2026-10-09T14:20:13.438372+00:00`.
- LeukemiaAttri: ejecución completa local `2026-10-09T20:06:45.762527+00:00`.
- LeukemiaAttri, actualización parcial: `2026-10-12T00:25:15.363564+00:00`
  (11 de octubre en Ciudad de México); reutilizó tablas, sin nueva inspección.
- Informe: elaborado con esas tablas y salidas; su copia para Git retira ejemplos
  visuales y dos captions completos. Las cifras y el análisis se conservan.

**No se ejecutaron los EDA durante este portado.** La compilación sintáctica,
validación del formato del notebook y pruebas de configuración no son una ejecución
del análisis ni una validación clínica. Las salidas agregadas retenidas proceden
de las ejecuciones anteriores, aunque la configuración de rutas ahora sea distinta.
Los contadores de ejecución se conservan solo donde hay salidas retenidas.

WBCAtt conserva salidas de las celdas originales 1, 5 y 19; LeukemiaAttri, de las
celdas 1, 13, 23 y 33 (índices desde cero; algunos outputs de esas celdas se omiten).
Se retiraron las otras salidas para evitar publicar microfotografías, captions
completos, muestras de registros y rutas personales. El código para generarlas
permanece; volver a ejecutar produce resultados privados que requieren otra revisión
antes de compartir. Las referencias del texto a tablas/ejemplos no visibles deben
leerse con esta aclaración, no como evidencia de una ejecución nueva.

## Datos esperados y configuración

Los datos no se distribuyen. Conservar el árbol de la investigación, por ejemplo:

```text
<raíz de datos externa>/
  Dataset/
    WBCAtt/pbc_attr_v1_{train,val,test}.csv
    captions/captions.csv
    LeukemiaAttri/
      LeukemiaAttri_Dataset-20260907T175427Z-1-001/
      LeukemiaAttri_Dataset-20260907T175427Z-1-002/
      LeukemiaAttri_Dataset-20260907T175427Z-1-003/
      LeukemiaAttri_Dataset-20260907T175427Z-1-004/
      LeukemiaAttri_Dataset-20260907T175427Z-1-005/
  bloodcells_dataset/<ocho carpetas PBC>/*.jpg
```

Cada fragmento conserva su interior `LeukemiaAttri_Dataset/<subconjunto>/...`,
JSON sueltos y ZIP internos de etiquetas. No fusionarlos, renombrarlos ni agregar
las copias Dataset Andres o ZIP de respaldo al alcance del lector.

Desde la raíz del repositorio, configurar antes de iniciar el kernel (ejemplo
PowerShell; sustituir rutas por ubicaciones propias):

```powershell
$env:HEMVLM_EDA_DATASET_DIR = 'D:/datos/HemVLM/Dataset'
$env:HEMVLM_EDA_PBC_DIR = 'D:/datos/HemVLM/bloodcells_dataset'
# Opcionales si las ubicaciones difieren del árbol anterior:
# $env:HEMVLM_EDA_WBC_DIR = 'D:/datos/WBCAtt'
# $env:HEMVLM_EDA_LEUKEMIA_DIR = 'D:/datos/LeukemiaAttri'
# $env:HEMVLM_EDA_CAPTIONS_CSV = 'D:/datos/captions/captions.csv'
```

Sin variables, se espera `data_local/Dataset` y `data_local/bloodcells_dataset`
en el repositorio; no se crean ni copian datasets automáticamente. Todas las rutas
relativas se resuelven desde la raíz Git. La salida predeterminada es
`artifacts/eda`, ignorada por Git; se puede configurar con `HEMVLM_EDA_OUTPUT_DIR`.
La salida debe quedar fuera de las fuentes y de notebooks/docs. Si se elige otro
directorio dentro del repositorio, añadirlo al ignore antes de ejecutar.

## Reproducción completa (no realizada durante este portado)

Crear un entorno independiente. Las versiones registradas utilizan Python 3.14.6;
la compatibilidad con otro entorno no fue comprobada. No se requiere GPU ni las
dependencias de entrenamiento del requirements.txt principal.

```powershell
python -m venv .venv-eda
.venv-eda/Scripts/Activate.ps1
python -m pip install -r requirements-eda.txt
python -m pip install jupyterlab
python -m jupyterlab
```

Abrir los notebooks desde la raíz del repositorio o desde `notebooks/eda`.
Antes de ejecutar, guardar una copia de trabajo bajo `artifacts/eda`; así no se
guardan salidas privadas sobre la copia destinada a Git. Reiniciar kernel y
ejecutar todas las celdas, en orden. Cada notebook genera su propia SQLite,
figuras y diagnósticos en `artifacts/eda/resultados_wbcatt` o
`artifacts/eda/resultados_leukemiaattri`. La sección 12 de LeukemiaAttri consume
las tablas producidas por las secciones anteriores. Las nuevas ejecuciones
reemplazan derivados, no datasets. Se necesita espacio para ambas SQLite
(aproximadamente 482,2 MiB en conjunto en la instantánea local), figuras y salidas.

No ejecutar scripts de entrenamiento, construcción de captions, extracción de
recortes o limpieza para reproducir estos EDA. Los notebooks contienen el análisis;
el único módulo auxiliar requerido es `eda_paths.py`.

### Repetir solo el desglose de LeukemiaAttri

Se necesita una **copia** de la SQLite de la ejecución completa compatible, dentro
de `artifacts/eda/resultados_leukemiaattri`. No está incluida en Git. Copiarla con
un nombre nuevo o en un destino vacío para conservar la fuente. Reiniciar kernel
y ejecutar exclusivamente las celdas de la sección 12, en orden. Esa sección
valida los nombres/columnas necesarios y la ejecución de origen; no inspecciona
imágenes. `HEMVLM_EDA_REVISION_DB` admite otra copia dentro del directorio de
salida; la configuración rechaza una SQLite situada fuera de este directorio.
La sección escribe los desgloses en esa copia. No afirmar que esto constituye
una ejecución completa ni una inspección actual de los originales.

## Archivos que permanecen locales

| Fuente bajo `reinicio_eda/` | Bytes | Motivo |
|---|---:|---|
| `resultados_wbcatt/tablas_eda.sqlite` | 99 602 432 | 95 MiB: detalle por imagen, captions completos y rutas; regenerable |
| `resultados_leukemiaattri/tablas_eda.sqlite` | 406 040 576 | 387,2 MiB: registros originales derivados, captions, rutas y checkpoint de 25 790 filas |
| `resultados_leukemiaattri/04_ejemplos_campos.pdf` | 6 084 314 | Microfotografías y cajas; permiso de redistribución del dataset pendiente |
| `01_EDA_WBCAtt.ipynb.bak` | 2 613 171 | Respaldo; no fuente de resultados vigentes |
| `02_EDA_LeukemiaAttri.ipynb.bak` | 10 410 133 | Respaldo; no fuente de resultados vigentes |

Cada SQLite contiene 68 652 filas del diagnóstico completo de captions, además
de subconjuntos seleccionados y tablas derivadas. No se incluyen por volumen y
por falta de permiso claro para redistribuir captions. No se sustituyen por un
volcado masivo CSV. El checkpoint interno no se cuenta como datos adicionales.

Las versiones originales de notebooks, inventario e informes permanecen intactas
en `reinicio_eda`; las copias aquí adaptadas difieren de ellas expresamente. No se
copiaron imágenes originales, CSV fuente, JSON/TXT fuente, ZIP, entornos,
credenciales ni temporales. Las seis PNG se copiaron sin modificar sus bytes.
Las dos microfotografías compuestas del informe y los captions completos se
retiraron de PDF y DOCX, incluidos sus objetos de medios; no se ocultaron solo
visualmente. El PDF conserva 18 páginas y señala las omisiones.

## Límites científicos que se conservan

WBCAtt: solo cuatro de once atributos tienen reglas comparativas; 1 261 imágenes
con contradicciones candidatas no son errores confirmados. LeukemiaAttri: 21
imágenes sin referencia/registros JSON (dos con TXT, 19 sin ambas representaciones);
cinco grupos duplicados con conjuntos distintos y diez no comparables, ninguno
con concordancia comprobada. Las 3 205 señales TXT dependen de la hipótesis de
coordenadas normalizadas. Un caption de campo no se asigna automáticamente a
una célula. Los hashes no descartan relaciones por paciente. No se traducen
códigos desconocidos ni se presenta un dataset final validado.

Consultar [REDISTRIBUCION.md](REDISTRIBUCION.md) antes de ampliar las inclusiones.
