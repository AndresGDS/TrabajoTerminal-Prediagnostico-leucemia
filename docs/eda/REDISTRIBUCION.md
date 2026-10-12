# Redistribución y atribución del aporte EDA

Revisión documental: 12 de octubre de 2026. Preparación local, sin publicación.
Esta revisión identifica evidencia y asuntos pendientes; no reemplaza un permiso
de los titulares ni extiende automáticamente la licencia de un código a sus datos.

| Componente | Evidencia consultada | Decisión para esta rama |
|---|---|---|
| PBC | Acevedo et al., Mendeley Data V1, DOI 10.17632/snkd93bnjr.1: CC BY 4.0 | No se incluyen originales ni microfotografías incrustadas; gráficas agregadas sí |
| WBCAtt | Repositorio apple2373/wbcatt: LICENSE MIT; submission/README enlaza PBC y los tres CSV | No se copian CSV ni código ajeno. Se incluyen código propio del EDA y estadísticas agregadas, con atribución |
| LeukemiaAttri | README oficial con enlace de descarga; no se encontró una autorización explícita de redistribución del dataset en las páginas revisadas | No se incluyen imágenes, etiquetas completas ni PDF de campos. Gráficas agregadas del análisis propio sí; derechos del dataset pendientes |
| Captions de Arturo | Usuario confirmó extracción desde el ZIP recibido; no se aportó licencia | No se incluyen CSV, SQLite ni captions íntegros en salidas, inventario o informes. Se conserva código de reglas, frecuencias y limitaciones |
| Repositorio destino y contribución | No se encontró LICENSE en el checkout base | No se añade una licencia elegida por suposición. Acordar con responsables la licencia de código/documentación antes de una distribución posterior |

## Fuentes de la revisión

- PBC: Acevedo, Andrea; Merino, Anna; Alférez, Santiago; Molina, Ángel; Boldú,
  Laura; Rodellar, José (2020). *A dataset for microscopic peripheral blood cell
  images for development of automatic recognition systems*. Mendeley Data, V1.
  https://data.mendeley.com/datasets/snkd93bnjr/1
  Licencia enlazada: https://creativecommons.org/licenses/by/4.0/
- WBCAtt: Tsutsui, Satoshi; Pang, Winnie; Wen, Bihan (2023). *WBCAtt: A White
  Blood Cell Dataset Annotated with Detailed Morphological Attributes*. NeurIPS.
  https://github.com/apple2373/wbcatt
  https://github.com/apple2373/wbcatt/blob/main/LICENSE
  https://github.com/apple2373/wbcatt/blob/main/submission/README.md
- LeukemiaAttri: Rehman, Abdul; Meraj, Talha; Minhas, Aiman Mahmood; Imran,
  Ayisha; Ali, Mohsen; Sultani, Waqas (2024). *A large-scale multi domain leukemia
  dataset for the white blood cells detection with morphological attributes for
  explainability*. MICCAI.
  https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024

No se descargaron datasets para esta revisión ni se verificó igualdad byte a byte
entre los archivos locales y una distribución pública de referencia. La licencia
PBC identificada no se usa como sustituto de comprobar esa procedencia. El README
de LeukemiaAttri describe formatos YOLO, pero esta revisión de permisos no cambia
ni valida la interpretación de los archivos locales en el EDA.

## Contenido derivado conservado y omitido

Las seis PNG son gráficos propios de frecuencias agregadas, no reproducciones de
microfotografías ni un volcado de anotaciones. Los notebooks retienen solo salidas
agregadas seleccionadas; código y reglas se conservan. El informe para Git omite
las figuras de microfotografías 4 y 8 y dos captions completos; todas sus cifras
se mantienen. El inventario conserva las correcciones históricas y omite el
contenido textual de caption/caption_with_image_token en su muestra JSON.

El PDF de campos completo sigue local por falta de autorización explícita
verificada. Antes de incorporarlo se necesita documentar permiso de imágenes y
anotaciones LeukemiaAttri. Para captions se necesita confirmar permiso de Arturo
y, cuando corresponda, de otros titulares. Que un enlace sea público, que el
usuario haya recibido un ZIP o que una fuente tenga un artículo no demuestra
por sí solo permiso para redistribuir todo su contenido.

Las salidas futuras pueden volver a incrustar contenido omitido: revisarlas antes
de un commit. Nada en esta preparación autoriza subir los originales, cambiar
sus licencias ni publicar datos privados.
