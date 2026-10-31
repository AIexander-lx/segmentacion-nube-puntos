# Segmentación de una nube de puntos de fábrica con clustering

Actividad grupal del curso de **Machine Learning** (UTEC, ciclo 2026-2), realizada por los grupos de Proyecto 1 y 2.

## Grupo

- **ID del grupo:** _(completar)_
- Zheng Chan, Enrique
- Limaco Porras, Requelmy
- León Pantaleón, Jobeth Alexander
- Huamán Huamaní, Josué Yeremi

## La actividad

| | |
|---|---|
| **Disponible** | 29 de octubre de 2026, 7:00 am |
| **Fecha límite** | **20 de noviembre de 2026, 11:59 pm** |
| **Intentos** | Ilimitados |
| **Quién entrega** | Solo **un** integrante del grupo |

**Dato de entrada.** Una nube de puntos 3D en formato PLY correspondiente a una fábrica (`FARO_Scan_Full_Sub1_down3_with_normals.ply`, ≈1,65 M de puntos). Cada punto tiene posición (x, y, z), color (RGB) y normal (nx, ny, nz).

**Objetivo.** Segmentar los objetos presentes (columnas, tuberías, pisos, techos, vigas, personas, vehículos, entre otros). La consigna recomienda **combinar algoritmos de clustering en múltiples etapas**, aprovechando los distintos atributos disponibles.

**Resultado esperado.** Exportar la nube de puntos con un color por punto que represente el segmento identificado.

**Herramientas sugeridas.** Open3D (Python) para leer y exportar nubes de puntos; MeshLab y CloudCompare para visualizar y generar capturas.

### Entregables

1. **Reporte en PDF** con el procedimiento propuesto y capturas de los resultados. Debe incluir el **identificador del grupo** y los **nombres completos de todos los integrantes**.
2. **Código fuente**, en un ZIP o mediante un enlace a un repositorio incluido en el reporte.
3. **Nube de puntos coloreada** según las clases encontradas, en formato **PLY**.

## División del trabajo por roles

El trabajo se reparte en cuatro roles de carga similar. Cada rol tiene un entregable de código, una sección del reporte y capturas propias, y cada integrante debe poder explicar su parte y entender las de los demás.

> **Cada integrante se asigna un rol** escribiendo su nombre en la columna "Responsable".

| Rol | Responsable | Responsabilidades | Aporta al código | Aporta al reporte |
|---|---|---|---|---|
| **1. Datos y preprocesamiento** | _(asignar)_ | Descargar y leer el PLY; explorar posiciones, colores y normales; submuestreo (voxel); limpieza de ruido y valores atípicos; escalado y construcción de las variables que usarán las etapas de clustering; decidir cómo propagar etiquetas a la nube completa. | `src/segmentacion/io.py`, preprocesamiento y características | Descripción de los datos, exploración y preprocesamiento |
| **2. Etapa 1: estructura global** | _(asignar)_ | Separar las superficies grandes (piso, techo, paredes) usando altura, normales y planaridad; definir los criterios de corte; dejar el resto de puntos listo para la etapa 2. | Módulo de la etapa 1 | Metodología de la etapa 1 y justificación de decisiones |
| **3. Etapa 2: objetos** | _(asignar)_ | Segmentar columnas, tuberías, vigas, personas y vehículos sobre los puntos restantes; probar y comparar algoritmos de clustering (por ejemplo DBSCAN/HDBSCAN, k-means, mezclas gaussianas, jerárquico) y sus hiperparámetros; refinar los segmentos. | Módulo de la etapa 2 | Metodología de la etapa 2, comparación de algoritmos |
| **4. Integración, evaluación y reporte** | _(asignar)_ | Unir las etapas en un pipeline reproducible (un solo script); asignar los colores por clase y exportar el PLY; generar capturas en CloudCompare/MeshLab; evaluación y discusión de resultados y limitaciones; armar el PDF final y verificar que cumple la consigna. | `scripts/` (pipeline), exportación del PLY | Integración del reporte, resultados, discusión y conclusiones |

### Responsabilidades compartidas

- Revisar los avances de los demás antes de integrar.
- Probar que el repositorio se ejecuta desde cero siguiendo este README.
- Todos participan en el reporte con su sección y revisan el documento completo.
- **Quién entrega en Canvas:** _(definir; solo uno del grupo)_.

## Datos

El PLY no se incluye en el repositorio por su tamaño. Descárgalo desde el enlace de la actividad en Canvas (Google Drive) y colócalo en `data/raw/`.

## Entorno

Las dependencias están declaradas en `pyproject.toml` y fijadas en `uv.lock`, con [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run jupyter lab
```

## Estructura del repositorio

- `src/segmentacion/`: código reutilizable.
- `notebooks/`: exploración y experimentos.
- `scripts/`: pipeline completo reproducible.
- `data/raw/`: nube de puntos original (no versionada).
- `outputs/`: PLY coloreado y figuras.
- `report/`: reporte en PDF.
