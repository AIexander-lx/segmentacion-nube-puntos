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
| **1. Datos y preprocesamiento** | Alexander | Descargar y leer el PLY; explorar posiciones, colores y normales; submuestreo (voxel); limpieza de ruido y valores atípicos; escalado y construcción de las variables que usarán las etapas de clustering; decidir cómo propagar etiquetas a la nube completa. | `src/segmentacion/io.py`, preprocesamiento y características | Descripción de los datos, exploración y preprocesamiento |
| **2. Etapa 1: estructura global** | _(asignar)_ | Separar las superficies grandes (piso, techo, paredes) usando altura, normales y planaridad; definir los criterios de corte; dejar el resto de puntos listo para la etapa 2. | Módulo de la etapa 1 | Metodología de la etapa 1 y justificación de decisiones |
| **3. Etapa 2: objetos** | Enrique | Segmentar columnas, tuberías, vigas, personas y vehículos sobre los puntos restantes; probar y comparar algoritmos de clustering (por ejemplo DBSCAN/HDBSCAN, k-means, mezclas gaussianas, jerárquico) y sus hiperparámetros; refinar los segmentos. | Módulo de la etapa 2 | Metodología de la etapa 2, comparación de algoritmos |
| **4. Integración, evaluación y reporte** | _(asignar)_ | Unir las etapas en un pipeline reproducible (un solo script); asignar los colores por clase y exportar el PLY; generar capturas en CloudCompare/MeshLab; evaluación y discusión de resultados y limitaciones; armar el PDF final y verificar que cumple la consigna. | `scripts/` (pipeline), exportación del PLY | Integración del reporte, resultados, discusión y conclusiones |

### Responsabilidades compartidas

- Revisar los avances de los demás antes de integrar.
- Probar que el repositorio se ejecuta desde cero siguiendo este README.
- Todos participan en el reporte con su sección y revisan el documento completo.
- **Quién entrega en Canvas:** _(definir; solo uno del grupo)_.

## Tareas por rol

Marca cada tarea cambiando `[ ]` por `[x]` cuando esté lista y sube el cambio con un commit. Cada rol trabaja en su propia rama (por ejemplo `rol1-datos`) y se integra a `main` con un pull request revisado por al menos otro integrante.

### Rol 1: Datos y preprocesamiento
- [ ] Descargar el PLY y colocarlo en `data/raw/`.
- [ ] Cargar la nube con Open3D y confirmar número de puntos, rangos de x, y, z, colores y normales.
- [ ] Explorar la nube: histogramas de altura, de normales y de color; vistas desde arriba y de costado.
- [ ] Aplicar submuestreo por voxel y elegir el tamaño del voxel con una justificación.
- [ ] Limpiar ruido y valores atípicos (por ejemplo, filtro estadístico o por radio).
- [ ] Construir y escalar las variables para clustering (posición, color, normales; las unidades importan).
- [ ] Guardar la nube procesada (con sus normales) para que los demás roles la usen, y subir una función que la cargue.
- [ ] Definir cómo propagar las etiquetas de la nube submuestreada a la nube completa.
- [ ] Redactar la sección de datos y preprocesamiento del reporte.

### Rol 2: Etapa 1, estructura global
- [ ] Definir con qué atributos se separan piso, techo y paredes (altura, normales, planaridad).
- [ ] Segmentar el piso.
- [ ] Segmentar el techo.
- [ ] Segmentar las paredes y otras superficies grandes.
- [ ] Probar al menos dos enfoques (por ejemplo, un método basado en planos y uno de clustering) y comparar.
- [ ] Entregar a la etapa 2 los puntos restantes con sus etiquetas de la etapa 1.
- [ ] Capturas de la etapa 1 para el reporte.
- [ ] Redactar la metodología de la etapa 1 y las decisiones tomadas.

### Rol 3: Etapa 2, objetos
- [ ] Recibir los puntos restantes de la etapa 1.
- [ ] Probar al menos tres algoritmos de clustering (por ejemplo DBSCAN o HDBSCAN, k-means, mezclas gaussianas o jerárquico).
- [ ] Ajustar hiperparámetros (por ejemplo `eps`, `min_samples`, número de clusters) y justificar la elección.
- [ ] Separar columnas y vigas.
- [ ] Separar tuberías.
- [ ] Separar personas y vehículos.
- [ ] Refinar segmentos (unir fragmentos, descartar ruido).
- [ ] Capturas de la etapa 2 para el reporte.
- [ ] Redactar la metodología de la etapa 2 y la comparación de algoritmos.

### Rol 4: Integración, evaluación y reporte
- [ ] Crear el esqueleto del reporte (PDF) con el ID del grupo y los nombres completos de los cuatro integrantes.
- [ ] Preparar la función que exporta el PLY coloreado según la clase.
- [ ] Unir las etapas en un solo script que reproduce todo desde el PLY original.
- [ ] Verificar que el repositorio se ejecuta desde cero siguiendo este README.
- [ ] Generar capturas en CloudCompare o MeshLab (vista general y por clase).
- [ ] Evaluación de los resultados, discusión, limitaciones y conclusiones.
- [ ] Integrar las secciones de todos los roles en el reporte y revisar el documento completo.
- [ ] Verificar los tres entregables: PDF, código (ZIP o enlace al repositorio incluido en el reporte) y PLY coloreado.
- [ ] Entregar en Canvas (solo un integrante), antes del **20 de noviembre, 11:59 pm**.

### Tareas de todos
- [ ] Asignarse un rol en la tabla de arriba.
- [ ] Completar el ID del grupo en este README y en el reporte.
- [ ] Acordar quién entrega en Canvas.
- [ ] Clonar el repositorio y configurar el entorno con `uv sync`.
- [ ] Revisar el reporte final antes de entregar.

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
