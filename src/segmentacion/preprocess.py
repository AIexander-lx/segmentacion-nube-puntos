"""Preprocesamiento de la nube de puntos (Rol 1): submuestreo, limpieza y variables.

Flujo:  PLY -> voxel_downsample -> remove_outliers -> local_geometry -> build_features
Los vecindarios se calculan con cKDTree (SciPy), sin depender de Open3D.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import PointCloud


def voxel_downsample(pc: PointCloud, voxel: float) -> tuple[PointCloud, np.ndarray]:
    """Agrupa los puntos en una grilla de lado `voxel` (m) y promedia cada celda.

    Devuelve la nube submuestreada y `inverse` (n_original,), que indica a qué
    voxel pertenece cada punto original; sirve para propagar etiquetas al final.
    """
    xyz = pc.xyz.astype(np.float64)
    ijk = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    dims = ijk.max(0) + 1
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, first, inverse, counts = np.unique(lin, return_index=True, return_inverse=True, return_counts=True)
    m = len(counts)

    def mean_of(a: np.ndarray) -> np.ndarray:
        out = np.zeros((m, a.shape[1]))
        for c in range(a.shape[1]):
            out[:, c] = np.bincount(inverse, weights=a[:, c], minlength=m) / counts
        return out

    nrm = mean_of(pc.normals.astype(np.float64))
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-6                      # normales opuestas que se anulan
    nrm[bad] = pc.normals[first[bad]]            # (puede ser 0 si la fuente no la trae)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm /= np.where(norm > 0, norm, 1.0)         # las normales nulas se recalculan después
    out = PointCloud(
        mean_of(xyz).astype(np.float32),
        np.clip(np.rint(mean_of(pc.rgb.astype(np.float64))), 0, 255).astype(np.uint8),
        nrm.astype(np.float32),
    )
    return out, inverse.astype(np.int64)


def outlier_mask(xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """True para los puntos con al menos `min_neighbors` vecinos dentro de `radius` (m)."""
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True) - 1  # sin contarse
    return counts >= min_neighbors


def local_geometry(xyz: np.ndarray, k: int = 20) -> np.ndarray:
    """Forma local con PCA sobre los k vecinos (lectura 11).

    Devuelve (n,3): linealidad, planaridad y esfericidad, a partir de los
    autovalores l1>=l2>=l3 de la covarianza local:
      linealidad=(l1-l2)/l1  (tuberías, vigas)  planaridad=(l2-l3)/l1  (pisos, paredes)
      esfericidad=l3/l1      (volúmenes, ruido)
    """
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)
    out = np.empty((len(xyz), 3), dtype=np.float32)
    step = 100_000
    for s in range(0, len(xyz), step):
        nb = xyz[idx[s:s + step]]                      # (b,k,3)
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", nb, nb) / k
        ev = np.linalg.eigvalsh(cov)[:, ::-1].clip(min=1e-12)   # l1>=l2>=l3
        out[s:s + step, 0] = (ev[:, 0] - ev[:, 1]) / ev[:, 0]
        out[s:s + step, 1] = (ev[:, 1] - ev[:, 2]) / ev[:, 0]
        out[s:s + step, 2] = ev[:, 2] / ev[:, 0]
    return out


def estimate_normals(xyz: np.ndarray, idx: np.ndarray, k: int = 20) -> np.ndarray:
    """Normales por PCA local (autovector del menor autovalor) para los puntos `idx`.

    Se usa para rellenar las normales nulas (0,0,0) que trae la fuente. El signo
    de una normal estimada es ambiguo; conviene usar |nz| en lugar de nz.
    """
    tree = cKDTree(xyz)
    _, nb_idx = tree.query(xyz[idx], k=k)
    nb = xyz[nb_idx]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("bki,bkj->bij", nb, nb) / k
    _, vec = np.linalg.eigh(cov)                  # autovalores ascendentes
    return vec[:, :, 0].astype(np.float32)


FEATURE_NAMES = [
    "x", "y", "z", "nx", "ny", "nz", "abs_nz",
    "r", "g", "b", "linealidad", "planaridad", "esfericidad",
]


def build_features(pc: PointCloud, geometry: np.ndarray) -> np.ndarray:
    """Matriz (n,13) sin escalar, en el orden de FEATURE_NAMES (color en [0,1])."""
    return np.column_stack([
        pc.xyz, pc.normals, np.abs(pc.normals[:, 2]), pc.rgb / 255.0, geometry,
    ]).astype(np.float32)


def standardize(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza por columna (media 0, desviación 1). Devuelve (F_esc, media, desv)."""
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    return ((F - mu) / sd).astype(np.float32), mu, sd


def propagate_labels(labels_kept: np.ndarray, kept_to_voxel: np.ndarray,
                     inverse: np.ndarray, n_voxels: int, unlabeled: int = -1) -> np.ndarray:
    """Lleva las etiquetas de la nube procesada a TODOS los puntos originales.

    labels_kept: (n_kept,) etiqueta por punto procesado.
    kept_to_voxel: (n_kept,) índice del voxel de cada punto procesado.
    inverse: (n_original,) voxel de cada punto original (de voxel_downsample).
    Los puntos cuyo voxel fue descartado como ruido reciben `unlabeled`.
    """
    per_voxel = np.full(n_voxels, unlabeled, dtype=np.int64)
    per_voxel[kept_to_voxel] = labels_kept
    return per_voxel[inverse]
