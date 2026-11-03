"""Extracción iterativa de planos geométricos con RANSAC de Open3D."""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import open3d as o3d

def extract_planes(points: np.ndarray, distance_threshold: float=0.05, iterations: int=500, min_points: int=3000, max_planes: int=6, seed: int=42, max_ransac_points: int=100000, vertical_axis: int|None=None) -> tuple[np.ndarray,list[dict[str,Any]],float]:
    """Devuelve etiquetas por punto (-1 sin plano), modelos y tiempo RANSAC.

    RANSAC se ejecuta en muestra acotada; cada modelo se valida contra todos
    los puntos restantes. vertical_axis se registra como metadato y no filtra modelos.
    """
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3: raise ValueError("points debe tener forma (n,3)")
    if distance_threshold<=0 or iterations<1 or min_points<3 or max_planes<0 or max_ransac_points<3: raise ValueError("Parámetros RANSAC inválidos")
    if vertical_axis is not None and vertical_axis not in (0,1,2): raise ValueError("vertical_axis debe ser 0, 1, 2 o None")
    labels=np.full(len(p),-1,dtype=np.int32); remaining=np.arange(len(p)); models=[]; elapsed=0.0
    o3d.utility.random.seed(int(seed))
    for _ in range(max_planes):
        if len(remaining)<min_points: break
        rng=np.random.default_rng(seed+len(models)); candidates=remaining if len(remaining)<=max_ransac_points else np.sort(rng.choice(remaining,max_ransac_points,replace=False))
        pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[candidates]))
        start=time.perf_counter(); plane,_=pc.segment_plane(distance_threshold,3,iterations); elapsed+=time.perf_counter()-start
        a,b,c,d=map(float,plane); norm=float(np.linalg.norm([a,b,c]))
        if norm==0: break
        residual=np.abs(p[remaining]@np.array([a,b,c])+d)/norm
        inliers=remaining[residual<=distance_threshold]
        if len(inliers)<min_points: break
        pid=len(models); labels[inliers]=pid
        xyz=p[inliers]; spans=np.ptp(xyz,axis=0)
        models.append({"id":pid,"equation":[a,b,c,d],"normal":[a/norm,b/norm,c/norm],"point_count":int(len(inliers)),"extent_xyz":spans.tolist(),"vertical_axis_configured":vertical_axis})
        mask=residual>distance_threshold; remaining=remaining[mask]
    return labels,models,elapsed
