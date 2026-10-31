"""Lectura y escritura de nubes de puntos PLY (posición, color RGB y normales)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_ply(path: str | Path) -> o3d.geometry.PointCloud:
    """Carga un PLY y verifica que tenga puntos."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"La nube de puntos está vacía: {path}")
    return pcd


def save_labeled_ply(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    palette: np.ndarray,
    path: str | Path,
) -> None:
    """Guarda la nube coloreada por segmento.

    labels: (n,) enteros; -1 = ruido / sin clase.
    palette: (k, 3) en [0, 1]; la clase -1 se pinta de gris.
    """
    colors = np.full((len(labels), 3), 0.5)
    ok = labels >= 0
    colors[ok] = palette[labels[ok] % len(palette)]
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), out)
