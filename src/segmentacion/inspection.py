"""Estadísticas compactas para inspeccionar nubes de puntos Open3D."""
from __future__ import annotations

import numpy as np
import open3d as o3d


def inspect_cloud(cloud: o3d.geometry.PointCloud) -> dict:
    """Resume geometría, atributos inválidos y memoria aproximada."""
    points = np.asarray(cloud.points)
    n = len(points)
    finite_xyz = np.isfinite(points)
    valid_rows = finite_xyz.all(axis=1) if n else np.zeros(0, dtype=bool)
    xyz = {axis: _range(points[:, i], valid_rows) for i, axis in enumerate("xyz")}
    dims = {axis: (v["max"] - v["min"] if v["min"] is not None else None) for axis, v in xyz.items()}
    scene_axis = max(dims, key=lambda a: dims[a] if dims[a] is not None else -1) if n and valid_rows.any() else None
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    invalid = int((~finite_xyz).sum())
    color_stats = _stats(colors) if colors is not None else None
    normal_stats = _stats(normals) if normals is not None else None
    zero_normals = int(np.count_nonzero(np.linalg.norm(normals, axis=1) == 0)) if normals is not None else None
    bytes_est = points.nbytes + (colors.nbytes if colors is not None else 0) + (normals.nbytes if normals is not None else 0)
    return {"point_count": n, "xyz_ranges": xyz, "scene_dimensions": dims, "largest_extent_axis_candidate": scene_axis, "colors_rgb": color_stats, "normals": normal_stats, "invalid_values": {"xyz_nan_or_inf": invalid, "xyz_rows_non_finite": int((~valid_rows).sum()), "color_nan_or_inf": int((~np.isfinite(colors)).sum()) if colors is not None else None, "normal_nan_or_inf": int((~np.isfinite(normals)).sum()) if normals is not None else None}, "zero_normals": zero_normals, "memory_bytes_estimate": bytes_est}

def _range(values: np.ndarray, valid: np.ndarray) -> dict:
    v = values[valid,] if values.ndim == 1 else values[valid, :]
    return {"min": float(np.min(v)) if v.size else None, "max": float(np.max(v)) if v.size else None}

def _stats(values: np.ndarray) -> dict:
    valid = values[np.isfinite(values)]
    return {"min": float(valid.min()) if valid.size else None, "max": float(valid.max()) if valid.size else None, "mean": float(valid.mean()) if valid.size else None, "nan_or_inf": int((~np.isfinite(values)).sum())}
