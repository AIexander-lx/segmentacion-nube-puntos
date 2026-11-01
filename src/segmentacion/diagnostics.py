"""Diagnósticos geométricos exploratorios, sin etiquetas semánticas."""
from __future__ import annotations
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def geometric_diagnostics(cloud: o3d.geometry.PointCloud, sample_size: int = 30000, seed: int = 42, voxels: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)) -> dict:
    """Compara coordenadas, alineación normal, vecinos y voxelizado acotados."""
    pts = np.asarray(cloud.points); finite = np.isfinite(pts).all(axis=1); pts = pts[finite]
    n = len(pts)
    if n == 0: raise ValueError("Se requieren puntos XYZ finitos")
    dims = np.ptp(pts, axis=0)
    # Muestreo fijo limita KD-tree y mantiene reproducibilidad.
    rng = np.random.default_rng(seed); sample = pts[rng.choice(n, min(n, sample_size), replace=False)]
    tree = cKDTree(sample); distances, _ = tree.query(sample, k=2, workers=1)
    nearest = distances[:, 1]
    normal_report = None
    if cloud.has_normals():
        normals = np.asarray(cloud.normals)[finite]
        good = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-8)
        unit = normals[good] / np.linalg.norm(normals[good], axis=1, keepdims=True)
        fractions = np.mean(np.abs(unit) >= 0.9, axis=0) if len(unit) else np.zeros(3)
        normal_report = {"valid_nonzero_count": int(len(unit)), "near_axis_fraction_abs_component_ge_0_9": dict(zip("xyz", fractions.tolist())), "horizontal_candidate_axis": "xyz"[int(np.argmax(fractions))] if len(unit) else None, "interpretation": "La componente normal dominante sugiere planos perpendiculares a ese eje; requiere corroborar con coordenadas."}
    voxel_rows=[]
    for size in voxels:
        start=time.perf_counter(); down=cloud.voxel_down_sample(size); elapsed=time.perf_counter()-start
        count=len(down.points)
        voxel_rows.append({"voxel_size": size, "point_count": count, "retained_percent": 100*count/len(cloud.points), "elapsed_seconds": elapsed, "downsampled_xyz_memory_bytes": count*3*8})
    return {"finite_point_count": n, "coordinate_quantiles": {a:{str(q):float(np.quantile(pts[:,i],q)) for q in (0.01,0.25,0.5,0.75,0.99)} for i,a in enumerate("xyz")}, "extent":dict(zip("xyz",dims.tolist())), "bounding_box_volume":float(np.prod(dims)), "approx_density_points_per_cubic_unit":float(n/np.prod(dims)) if np.prod(dims)>0 else None, "neighbor_sample":{"sample_size":len(sample),"seed":seed,"distance_quantiles":{str(q):float(np.quantile(nearest,q)) for q in (0.1,0.5,0.9,0.99)}}, "normal_orientation":normal_report, "voxel_comparison":voxel_rows}
