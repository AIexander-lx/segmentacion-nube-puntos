# SEGMENTACIÓN DE NUBE DE PUNTOS 3D

> Proyecto académico de Machine Learning: análisis geométrico y segmentación exploratoria de una fábrica escaneada.

## 1. Descripción y objetivo

El proyecto explora cómo separar estructuras de una escena industrial representada como una nube PLY con posiciones XYZ, colores RGB y normales. La implementación combina preprocesamiento multirresolución, detección geométrica, agrupamiento espacial, descriptores de forma y reglas semánticas interpretables. Los resultados son propuestas experimentales y requieren revisión humana.

## 2. Características

- Visor web local WebGL para las resoluciones original, global y detalle.
- Vistas RGB, normales, instancias geométricas y propuestas semánticas.
- Segmentación geométrica con planos RANSAC y componentes espaciales limitadas.
- Refinamiento V3 experimental por tiles, regiones voxelizadas y compatibilidad de normales.
- Exportación de PLY, etiquetas NPZ, métricas JSON y figuras reproducibles.
- Reconstrucción localizada de superficies con Ball Pivoting y Poisson.
- Pruebas unitarias sobre datos sintéticos; no requieren la fábrica para la mayoría de los casos.

## 3. Captura de la aplicación

No hay una captura verificada del visor incluida en este release. Puedes guardar una desde el botón **Guardar captura** de la aplicación después de cargar los datos locales. Las figuras de experimentos y de la fábrica no se publican aquí porque el permiso de redistribución del dataset no está documentado.

## 4. Arquitectura y carpetas

```text
main.py                     Launcher del visor local
web/                        Interfaz HTML, CSS y WebGL
src/segmentacion/           Lectura, inspección, geometría y pipelines
scripts/                    CLI reproducibles de inspección y procesamiento
tests/                      Suite unittest con nubes sintéticas
data/raw/                   Ubicación recomendada para el PLY local
outputs/                    Resultados generados; ignorados por Git
notebooks/                  Espacio para exploración del curso
report/                     Espacio para el reporte académico
```

El servidor enlaza solo a `127.0.0.1`. El visor detecta datos y resultados disponibles y muestra avisos cuando faltan. No recalcula la segmentación al abrirse.

## 5. Requisitos

- Python `>=3.10,<3.13`; el entorno de desarrollo usado para esta versión fue Windows x64, Python 3.12.10.
- [uv](https://docs.astral.sh/uv/) para crear el entorno desde `uv.lock`.
- Las dependencias declaradas incluyen NumPy, SciPy, scikit-learn, Open3D, Matplotlib, pandas y JupyterLab.
- Open3D ofrece ruedas para plataformas comunes, pero la instalación debe comprobarse para la versión de Python y arquitectura de cada equipo. La validación completa de este checkout se realizó en Windows.

## 6. Instalar uv

En PowerShell, sigue el instalador oficial de uv o usa:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

En macOS o Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Consulta las instrucciones oficiales antes de ejecutar el instalador si necesitas revisar el script o elegir otro método.

## 7. Obtener la nube PLY

El PLY de la actividad (`FARO_Scan_Full_Sub1_down3_with_normals.ply`, aproximadamente 44.5 MB) se obtiene desde el recurso de Canvas del curso. No se incluye en Git porque el material recibido no contiene una licencia o permiso explícito de redistribución. Descárgalo con acceso al curso y colócalo en `data/raw/` con ese nombre. También se acepta en la raíz del proyecto.

El visor reconoce ambas ubicaciones. Los comandos de procesamiento aceptan una ruta local mediante `--input` o `--original`. El archivo no se mueve ni se copia automáticamente. Si no está disponible, el visor informa que falta la nube y el procesamiento se detiene con un mensaje explícito.

## 8. Instalar dependencias

Desde la raíz del repositorio:

```powershell
uv sync
```

```sh
uv sync
```

`uv.lock` fija las versiones resueltas. No se necesita copiar ni versionar `.venv/`.

## 9. Abrir la aplicación

El comando principal inicia el servidor en un puerto local libre y abre el navegador:

```powershell
uv run python main.py
```

```sh
uv run python main.py
```

Opciones útiles: `uv run python main.py --help` y `uv run python main.py --no-browser`. Detén el servidor con `Ctrl+C`.

## 10. Usar las pestañas

- **Original:** elige original, global o detalle; también puedes indicar una ruta a otro PLY local.
- **Segmentación:** elige RGB, normales, instancias o clases semánticas; selecciona V1/V3 cuando sus artefactos existan, filtra clases y cambia el nivel de detalle.
- **Reconstrucción:** muestra mallas localizadas disponibles y permite iniciar trabajos explícitos y acotados.
- En el visor puedes rotar con arrastre, acercar con la rueda, cambiar de cámara, encuadrar, guardar una captura PNG y exportar un PLY cuando esté disponible.
- Si el panel está plegado, pulsa **Mostrar herramientas**. La información técnica ampliada está bajo el visor.

Las clases se muestran con colores de instancia/clase propios del proyecto. Gris representa `unknown`; no equivale a `other`.

## 11. Procesar desde el PLY

### Opción rápida: un solo comando

Con el PLY en `data/raw/`, desde la raíz del repositorio:

```powershell
uv run python scripts/run_pipeline.py --input .\data\raw\FARO_Scan_Full_Sub1_down3_with_normals.ply
```

`scripts/run_pipeline.py` encadena en orden preprocesamiento, planos RANSAC y clustering, clasificación V1, componente grande y refinamiento V3 (los pasos detallados de abajo), con semilla 42. Deja el PLY coloreado por clase en `outputs/ply/fabrica_segmentada.ply` y el porcentaje de puntos por clase en `outputs/pipeline_summary.json`. Si ya existen resultados de una corrida anterior, añade `--force` para borrarlos y repetir. Tarda alrededor de un minuto (≈51 s en la corrida registrada). Con `--skip-v3` se detiene tras V1.

### Paso a paso

En PowerShell, después de colocar el archivo en `data/raw/`:

```powershell
$ply = ".\data\raw\FARO_Scan_Full_Sub1_down3_with_normals.ply"
uv run python scripts/preprocess_cloud.py --input $ply --global-voxel 0.10 --detail-voxel 0.05 --output-dir .\outputs\processed --seed 42
uv run python scripts/segment_cloud.py --original $ply --global-cloud .\outputs\processed\global.ply --mapping .\outputs\processed\global_mapping.npz --output-dir .\outputs\segmented\global_review_final --seed 42 --eps 0.25 --max-neighbors 64
uv run python scripts/classify_cloud.py --original $ply --global-cloud .\outputs\processed\global.ply --mapping .\outputs\processed\global_mapping.npz --labels .\outputs\segmented\global_review_final\labels.npz --planes-metrics .\outputs\segmented\global_review_final\metrics.json --output-dir .\outputs\semantic\global_heuristic_v1 --seed 42
```

Para ejecutar el refinamiento V3 se necesita primero la segmentación y clasificación anteriores. La selección de la componente grande se prepara con el script existente `scripts/analyze_giant.py`; ejecútalo una sola vez en un directorio de resultados limpio. Después:

```powershell
uv run python scripts/analyze_giant.py
uv run python scripts/segment_v3.py --original $ply --global-cloud .\outputs\processed\global.ply --mapping .\outputs\processed\global_mapping.npz --semantic-v1 .\outputs\semantic\global_heuristic_v1\semantic_labels.npz --geometry-v1 .\outputs\segmented\global_review_final\labels.npz --giant-indices .\outputs\experiments\giant_component_v3\giant_indices.npz --output-dir .\outputs\semantic\global_local_v3_release --tile-size 2.0 --halo 0.20 --voxel-size 0.20 --normal-angle 35 --min-region-points 20 --max-tile-points 5000 --max-edges 100000 --max-tiles 5000 --max-context-points 2000000 --max-global-points 400000 --max-original-points 2000000 --max-memory-mb 1024 --seed 42
```

Los CLI aceptan `--help`; por ejemplo, `uv run python scripts/segment_v3.py --help`. Los límites de memoria de los scripts son preflight por estimación y no una restricción impuesta por el sistema operativo. Cada corrida debe usar una carpeta de salida nueva; los scripts evitan sobrescribir resultados existentes.

Para ver una muestra o inspeccionar el PLY:

```powershell
uv run python scripts/inspect_cloud.py --input $ply --output .\outputs\metrics\inspection.json --figure --sample-size 100000 --seed 42
uv run python scripts/view_cloud.py --resolution global --color rgb --show
```

Para reconstruir una ROI —no toda la fábrica— usa una carpeta nueva; Ball Pivoting requiere normales y Poisson puede inferir o cerrar superficies sin observación directa:

```powershell
uv run python scripts/reconstruct_cloud.py --input .\outputs\processed\global.ply --mapping .\outputs\processed\global_mapping.npz --roi 10.9488 12.4488 2.8843 4.3843 1.1687 2.6688 --output-dir .\outputs\reconstruction\roi_run_001 --max-points 3000 --poisson-depth 5 --seed 42
```

Los límites, radios BPA y controles disponibles se describen en `uv run python scripts/reconstruct_cloud.py --help`.

## 12. Pruebas unitarias

```powershell
uv run python -m unittest discover -s tests -v
```

La suite utiliza nubes pequeñas y fixtures temporales. La versión preparada para este release tiene 72 pruebas aprobadas en el entorno de desarrollo registrado; los tiempos y el resultado pueden variar según el equipo.

## 13. Métodos utilizados

- **Open3D:** lectura y escritura PLY, estructuras de nube y reconstrucción localizada.
- **Voxelización:** reduce densidad y conserva mapeos entre puntos fuente y celdas.
- **RANSAC:** detecta planos dominantes geométricos; un plano no se considera automáticamente piso o pared.
- **Agrupamiento espacial:** componentes de vecindad con límites para controlar memoria; el ruido y los segmentos grandes pueden quedar sin asignar.
- **PCA y normales:** resumen orientación y forma local; se usan como evidencia geométrica.
- **Reglas semánticas:** proponen clases interpretables (floor, ceiling, wall, column, beam, pipe, other, unknown). Sus scores no son probabilidades.
- **BPA/Poisson:** reconstrucción de mallas solo sobre regiones pequeñas; las superficies generadas se consideran inferidas.

## 14. Resultados experimentales registrados

| Medida | Resultado |
|---|---:|
| Puntos del PLY original | 1,647,021 |
| Nube global (voxel 0.10) | 305,071 |
| Nube detalle (voxel 0.05) | 1,041,806 |
| V1 con clase propuesta | 12.796% de puntos originales |
| V1 `unknown` | 87.204% |
| V3 con clase propuesta | 42.806% de puntos originales |
| V3 `unknown` | 57.194% |
| Tiempo del pipeline completo (`run_pipeline.py`) | 50.8 s |

Distribución de clases del resultado final (V3), sobre los 1,647,021 puntos originales:

| Clase | Puntos | % |
|---|---:|---:|
| floor | 134,991 | 8.196 |
| ceiling | 155,897 | 9.465 |
| wall | 9,769 | 0.593 |
| column | 124,893 | 7.583 |
| beam | 247,040 | 14.999 |
| pipe | 26,582 | 1.614 |
| other | 5,845 | 0.355 |
| unknown | 942,004 | 57.194 |

En el resultado guardado, V3 propuso 494,260 puntos originales adicionales respecto a V1 y mantuvo intactas las etiquetas conocidas de V1. El tiempo observado corresponde a un entorno de ejecución concreto, no es una garantía para otros equipos. No se registró una medición de memoria pico válida en el JSON.

**Corrección respecto a la versión anterior de esta sección.** La versión previa reportaba techo 0 % y 65.442 % `unknown` en V3 (V1: 91.372 % `unknown`). La causa era que la altura relativa de cada región se calculaba con el máximo de z de la nube. Unos 14 puntos aislados llegan a z ≈ 14.5 m mientras el techo real está en z ≈ 9 m, de modo que el techo nunca superaba el umbral de altura relativa de 0.78 y no se reconocía. Ahora `scripts/classify_cloud.py` y `src/segmentacion/semantic_v3.py` calculan los límites de escena con los percentiles 0.1 y 99.9 de las coordenadas. El techo pasa de 0 % a 9.465 %, y `unknown` baja de 65.442 % a 57.194 %.

Estas cifras miden **cobertura de reglas heurísticas**, no calidad semántica. No hay anotaciones humanas completas; por tanto, no se reportan accuracy, precisión, recall ni IoU semántico. No se ha demostrado que V3 sea más correcto que V1.

## 15. Limitaciones

- El PLY debe obtenerse desde Canvas; no se comprobó permiso para redistribuirlo públicamente.
- No existe ground truth semántico humano para evaluar exactitud.
- El eje vertical Z es una hipótesis de trabajo, no una confirmación universal.
- La cobertura V3 aumenta las propuestas, pero no demuestra mayor precisión; fronteras de tiles e instancias siguen siendo experimentales.
- La reconstrucción puede completar huecos o conectar geometría no observada; siempre debe revisarse junto a los puntos.
- Más de la mitad de los puntos (57.194 %) queda como `unknown`, y el vocabulario de clases no incluye personas ni vehículos.
- Las reglas usan umbrales fijados a mano (p. ej. altura relativa del 22 % y 78 %); dependen de la hipótesis del eje vertical y de la densidad de la nube.
- El procesamiento de resolución global puede consumir memoria considerable. No se ejecuta automáticamente al abrir la aplicación.
- El clean install con `uv sync` no se volvió a ejecutar en esta sesión porque `uv` no estaba en PATH; la suite sí se ejecutó con el Python 3.12.10 del entorno `.venv` existente.

## 16. Autores y grupo

Actividad grupal del curso de Machine Learning de UTEC (ciclo 2026-2). El README previo registra estos integrantes:

- Zheng Chan, Enrique
- Limaco Porras, Requelmy
- León Pantaleón, Jobeth Alexander
- Huamán Huamaní, Josué Yeremi

El identificador del grupo no estaba consignado en el README de origen; debe completarse por el equipo conforme a la actividad. La autoría y el historial Git del repositorio original se conservan.

## 17. Integración en `JosueBranch`

Esta rama integra el sistema funcional del repositorio [Jhynfe/Actividad_3_ML](https://github.com/Jhynfe/Actividad_3_ML) dentro del repositorio [AIexander-lx/segmentacion-nube-puntos](https://github.com/AIexander-lx/segmentacion-nube-puntos). Se conserva el visor, los pipelines de segmentación, clasificación y reconstrucción, sus scripts y pruebas. El historial de ambos proyectos queda unido por un commit de merge.

También se conserva el trabajo previo de exploración y preprocesamiento de la rama de datos: `scripts/01_explorar.py`, `scripts/02_preprocesar.py`, `scripts/03_validar_preprocesamiento.py` y `src/segmentacion/preprocess.py`. Estos scripts leen el PLY binario little-endian que provee Canvas y guardan el resultado comprimido en `data/processed/nube_procesada.npz`. Para ejecutarlos:

```powershell
uv run python scripts/01_explorar.py
uv run python scripts/02_preprocesar.py --voxel 0.08
uv run python scripts/03_validar_preprocesamiento.py
```

`src/segmentacion/io.py` mantiene ambas interfaces de lectura: `load_ply`/`save_labeled_ply` para los pipelines Open3D, y `read_ply`/`write_ply`/`PointCloud` para el preprocesamiento ligero con NumPy. Open3D es dependencia principal porque la aplicación y los pipelines integrados la requieren.

## 18. Fuentes, referencias y atribución

- [Open3D: procesamiento de nubes y RANSAC](https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html)
- [Open3D: reconstrucción de superficies, Ball Pivoting y Poisson](https://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html)
- [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [uv: instalación](https://docs.astral.sh/uv/getting-started/installation/) y [sincronización del entorno](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Library of Congress: formato PLY](https://www.loc.gov/preservation/digital/formats/fdd/fdd000501.shtml)

El dataset se atribuye a la actividad del curso y a su recurso de Canvas; no se redistribuye desde este repositorio. El repositorio original no incluía un archivo `LICENSE`; este release no asigna una licencia nueva. Para reutilizar el código fuera del contexto del curso, solicita autorización a sus autores.
