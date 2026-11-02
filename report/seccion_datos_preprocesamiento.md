# Datos y preprocesamiento (borrador, Rol 1)

> Borrador para el reporte. Revisar, adaptar al estilo del grupo y verificar cada cifra ejecutando `scripts/01_explorar.py`, `scripts/02_preprocesar.py` y `scripts/03_validar_preprocesamiento.py`. Figuras: `outputs/figures/01_alturas_normales_color.png`, `02_vistas.png` y `03_variables_locales_y_ruido.png`.

## 1. Descripción de los datos

La actividad utiliza una nube de puntos 3D en formato PLY de una planta industrial (`FARO_Scan_Full_Sub1_down3_with_normals.ply`), capturada con un escáner láser FARO. El archivo contiene 1 647 021 puntos en formato binario. Cada punto tiene nueve atributos: posición (x, y, z, en metros), color (R, G, B, de 0 a 255) y normal (nx, ny, nz). No hay etiquetas de clase, por lo que el problema es de aprendizaje no supervisado.

La nube abarca aproximadamente 31 m en x (de −2,6 a 28,8), 35 m en y (de −11,9 a 23,0) y 16,8 m en z (de −2,3 a 14,5). El nombre del archivo indica que ya fue submuestreado: la distancia mediana al vecino más cercano es de 3,4 cm, con muy poca dispersión (percentil 10 de 3,2 cm y percentil 90 de 3,8 cm), es decir, la densidad es casi uniforme.

## 2. Exploración

- **Altura.** El histograma de z muestra un pico muy marcado cerca de −1,8 m (el piso) y una concentración entre 8,5 y 9,2 m (el techo). Entre ambos hay estructuras a distintas alturas, como racks de tuberías cerca de 3 m y entre 5 y 6 m. Existen unos pocos puntos aislados por encima de 10 m.
- **Normales.** Están muy concentradas en dos extremos: el 49,4 % tiene |nz| mayor que 0,9 (superficies horizontales como pisos, techos y losas) y el 29,0 % tiene |nz| menor que 0,2 (superficies verticales como columnas y paredes).
- **Color.** El brillo medio tiene una distribución amplia, con un máximo de puntos oscuros y otro de puntos saturados. El color real distingue, por ejemplo, tuberías verdes y rojas del piso claro y del techo oscuro, pero no es suficiente por sí solo para separar clases.
- **Estructura.** La vista en planta muestra una grilla regular de columnas y un anexo metálico a la derecha (x entre 22 y 28 m). Los perfiles confirman un techo con cerchas y varios niveles de estructuras.
- **Calidad.** Dos puntos de la fuente tienen la normal nula (0, 0, 0).

## 3. Submuestreo por voxel

Con 1,65 millones de puntos, algoritmos como DBSCAN o la matriz de vecinos resultan costosos. Se submuestrea con una grilla de voxeles de lado *v*: los puntos de cada celda se reemplazan por su promedio (posición y color) y por la normal promedio renormalizada. Se guarda además la correspondencia punto original → voxel para poder propagar las etiquetas al final.

| Tamaño de voxel | Puntos restantes | % del original |
|---|---|---|
| 0,03 m | 1 629 452 | 98,9 % |
| 0,05 m | 1 041 806 | 63,3 % |
| **0,08 m** | **459 991** | **27,9 %** |
| 0,10 m | 305 071 | 18,5 % |
| 0,15 m | 142 839 | 8,7 % |

Con 3 cm casi no se reduce nada, porque la nube ya tiene ese espaciado. Un voxel de 10 cm o más puede fusionar elementos delgados como tuberías pequeñas. Se eligió **8 cm**, que reduce la nube a menos de un tercio y mantiene los elementos delgados representados por varios voxeles. El valor es un parámetro (`--voxel`) y se puede ajustar si las etapas de clustering lo requieren.

## 4. Limpieza de ruido

Se descartan los voxeles aislados: un punto se conserva solo si tiene al menos 4 vecinos dentro de un radio de 0,25 m. Se eliminaron 1 006 voxeles (0,22 %), dispersos por la nube y no concentrados en una región. La figura `03_variables_locales_y_ruido.png` muestra su ubicación. Quedan **458 985 puntos**.

El filtro es deliberadamente suave para no eliminar objetos pequeños pero reales, como personas o elementos delgados; la mayoría de los puntos removidos son aislados.

## 5. Variables para clustering

Cada punto procesado se describe con 13 variables:

1. Posición: x, y, z.
2. Normal: nx, ny, nz, y su verticalidad |nz| (el signo de una normal es ambiguo, por lo que |nz| es más fiable).
3. Color: r, g, b, normalizados a [0, 1].
4. Forma local (PCA sobre los 20 vecinos más cercanos): a partir de los autovalores λ1 ≥ λ2 ≥ λ3 de la covarianza local se calculan la **linealidad** (λ1 − λ2)/λ1, que es alta en tuberías y vigas; la **planaridad** (λ2 − λ3)/λ1, alta en pisos, techos y paredes; y la **esfericidad** λ3/λ1, alta en volúmenes y ruido. Son una aplicación directa del análisis de componentes principales visto en clase.

Como el clustering se basa en distancias y las variables están en unidades distintas (metros, intensidades de color, valores adimensionales), todas se **estandarizan** (media 0 y desviación 1). Se guardan tanto la versión original como la estandarizada, junto con la media y la desviación usadas.

## 6. Propagación de etiquetas a la nube completa

El clustering se hace sobre la nube procesada. Para exportar el PLY final con todos los puntos, la etiqueta de cada punto original es la de su voxel; los puntos cuyo voxel fue descartado como ruido quedan sin etiqueta (−1). En una prueba de consistencia con una etiqueta artificial (z mayor o menor que 3 m), la etiqueta propagada coincidió con la directa en el 99,59 % de los puntos, y solo 1 308 puntos originales (0,08 %) quedaron sin etiqueta. Las diferencias se deben a puntos cercanos al borde de un voxel.

## 7. Resultado

El preprocesamiento deja una nube de 458 985 puntos con 13 variables por punto, guardada en `data/processed/nube_procesada.npz` y cargable con `load_processed()`. Los parámetros (voxel, radio, número mínimo de vecinos y k) están registrados en el archivo, de modo que el procedimiento es reproducible.
