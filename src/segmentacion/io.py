"""Lectura y escritura segura de nubes de puntos PLY mediante Open3D."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_ply(path: str | Path) -> o3d.geometry.PointCloud:
    """Carga un PLY existente y rechaza archivos vacíos o ilegibles."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el archivo PLY: {source}")
    pcd = o3d.io.read_point_cloud(str(source))
    if not pcd.has_points():
        raise ValueError(f"El PLY no contiene puntos legibles: {source}")
    return pcd


def save_labeled_ply(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    palette: np.ndarray,
    path: str | Path,
) -> Path:
    """Exporta una copia coloreada; nunca escribe sobre la nube de entrada."""
    target = Path(path).expanduser()
    if pcd.is_empty():
        raise ValueError("No se puede exportar una nube vacía")
    labels = np.asarray(labels)
    palette = np.asarray(palette, dtype=float)
    if labels.ndim != 1 or len(labels) != len(pcd.points):
        raise ValueError("labels debe tener una etiqueta por punto")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette debe tener forma (k, 3), con k > 0")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError("labels debe contener enteros >= -1")
    if not np.isfinite(palette).all() or np.any((palette < 0) | (palette > 1)):
        raise ValueError("palette debe ser finita y estar en [0, 1]")
    colors = np.full((len(labels), 3), 0.5)
    assigned = labels >= 0
    colors[assigned] = palette[labels[assigned] % len(palette)]
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() == Path.cwd().resolve() or (target.exists() and target.resolve() == Path.cwd().resolve()):
        raise ValueError("La ruta de salida debe ser un archivo, no un directorio")
    if not o3d.io.write_point_cloud(str(target), out):
        raise OSError(f"No se pudo escribir el PLY: {target}")
    return target
