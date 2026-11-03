"""Preprocesamiento voxelizado con correspondencia explícita a la nube original."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any
import numpy as np
import open3d as o3d

@dataclass
class VoxelResult:
    """Nube voxelizada y mapeos; -1 indica XYZ original inválido."""
    cloud: o3d.geometry.PointCloud
    original_to_voxel: np.ndarray
    voxel_to_representative: np.ndarray
    normal_valid: np.ndarray
    stats: dict[str, Any]

def validate_xyz(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """Devuelve máscara por fila con XYZ finito; no muta ni descarta la fuente."""
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("La nube debe tener puntos XYZ con forma (n, 3)")
    return np.isfinite(points).all(axis=1)

def _normalize_normals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza promedios válidos; deja cero cuando no hay orientación conocida."""
    lengths = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (lengths > 1e-12)
    out = np.zeros_like(values, dtype=np.float64)
    out[valid] = values[valid] / lengths[valid, None]
    return out, valid

def voxel_reduce(cloud: o3d.geometry.PointCloud, voxel_size: float) -> VoxelResult:
    """Agrupa por celdas voxel desde el mínimo XYZ y agrega atributos por media.

    Retorna un índice voxel por punto fuente y un punto fuente representativo
    (más cercano al centroide) por voxel. El cálculo usa unique/bincount, no NxN.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size debe ser un número positivo y finito")
    started = time.perf_counter()
    points = np.asarray(cloud.points)
    valid = validate_xyz(cloud)
    if not valid.any():
        raise ValueError("No hay puntos XYZ finitos para voxelizar")
    xyz = points[valid]
    origin = xyz.min(axis=0)
    keys = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    sizes = np.bincount(inverse, minlength=count).astype(np.int64)
    sums = np.column_stack([np.bincount(inverse, weights=xyz[:, axis], minlength=count) for axis in range(3)])
    centroids = sums / sizes[:, None]
    colors_out = None
    if cloud.has_colors():
        colors = np.asarray(cloud.colors)
        if len(colors) != len(points): raise ValueError("RGB debe tener una fila por punto")
        c = colors[valid]
        if not np.isfinite(c).all(): raise ValueError("RGB contiene valores no finitos")
        colors_out = np.column_stack([np.bincount(inverse, weights=c[:, axis], minlength=count) / sizes for axis in range(3)])
    normals_out = None; normal_valid = np.zeros(count, dtype=bool)
    if cloud.has_normals():
        normals = np.asarray(cloud.normals)
        if len(normals) != len(points): raise ValueError("Las normales deben tener una fila por punto")
        usable = np.isfinite(normals[valid]).all(axis=1) & (np.linalg.norm(normals[valid], axis=1) > 1e-12)
        nsums = np.column_stack([np.bincount(inverse[usable], weights=normals[valid][usable, axis], minlength=count) for axis in range(3)])
        normals_out, normal_valid = _normalize_normals(nsums)
    # Representante original próximo al centroide. np.minimum.at mantiene memoria lineal.
    d2 = np.einsum("ij,ij->i", xyz - centroids[inverse], xyz - centroids[inverse])
    representatives = np.full(count, len(points), dtype=np.int64)
    best = np.full(count, np.inf)
    np.minimum.at(best, inverse, d2)
    candidate = np.where(d2 == best[inverse], np.flatnonzero(valid), len(points))
    np.minimum.at(representatives, inverse, candidate)
    original_to_voxel = np.full(len(points), -1, dtype=np.int64)
    original_to_voxel[valid] = inverse
    reduced = o3d.geometry.PointCloud()
    reduced.points = o3d.utility.Vector3dVector(centroids)
    if colors_out is not None: reduced.colors = o3d.utility.Vector3dVector(colors_out)
    if normals_out is not None: reduced.normals = o3d.utility.Vector3dVector(normals_out)
    stats = {"input_points": int(len(points)), "valid_xyz_points": int(valid.sum()), "output_voxels": count, "retained_percent_valid": 100.0 * count / int(valid.sum()), "voxel_size": float(voxel_size), "elapsed_seconds": time.perf_counter() - started, "output_attribute_bytes": int(centroids.nbytes + (0 if colors_out is None else colors_out.nbytes) + (0 if normals_out is None else normals_out.nbytes)), "invalid_xyz_excluded": int((~valid).sum()), "voxels_without_valid_normal": int((~normal_valid).sum()) if normals_out is not None else None}
    return VoxelResult(reduced, original_to_voxel, representatives, normal_valid, stats)

def propagate_voxel_labels(labels: np.ndarray, original_to_voxel: np.ndarray) -> np.ndarray:
    """Propaga etiquetas de vóxel a original; puntos inválidos quedan en -1."""
    labels = np.asarray(labels)
    mapping = np.asarray(original_to_voxel)
    if labels.ndim != 1 or mapping.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels y mapping deben ser vectores; labels entero")
    valid = mapping >= 0
    if np.any(mapping[valid] >= len(labels)): raise ValueError("El mapeo refiere a un vóxel inexistente")
    result = np.full(len(mapping), -1, dtype=labels.dtype)
    result[valid] = labels[mapping[valid]]
    return result
