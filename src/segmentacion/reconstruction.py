"""Reconstrucción experimental de superficies sobre recortes manejables."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

@dataclass
class PreparedCloud:
    cloud: o3d.geometry.PointCloud
    source_indices: np.ndarray
    excluded_invalid_xyz: int
    excluded_invalid_normals: int
    orientation: str
    orientation_consistency_before: float|None
    orientation_consistency_after: float|None

def _consistency(normals: np.ndarray, points: np.ndarray, sample_limit: int=5000) -> float|None:
    if len(points)<2:return None
    ids=np.arange(len(points)) if len(points)<=sample_limit else np.random.default_rng(42).choice(len(points),sample_limit,replace=False)
    tree=cKDTree(points);_,idx=tree.query(points[ids],k=2,workers=1);a=normals[ids];b=normals[idx[:,1]]
    valid=(np.linalg.norm(a,axis=1)>0)&(np.linalg.norm(b,axis=1)>0)
    if not valid.any():return None
    dot=np.sum(a[valid]*b[valid],axis=1);return float(np.mean(dot>=0))

def prepare_oriented_cloud(cloud: o3d.geometry.PointCloud, *, orient: bool=True, neighbor_k: int=30) -> PreparedCloud:
    """Copies valid XYZ/normals, normalizes and optionally orients local normals.

    Points with invalid/zero normals are excluded, never assigned invented directions.
    """
    points=np.asarray(cloud.points);normals=np.asarray(cloud.normals) if cloud.has_normals() else np.zeros_like(points)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError("Cloud XYZ must have shape (n,3)")
    if not cloud.has_normals() or len(normals)!=len(points):raise ValueError("Reconstruction requires one normal per point")
    valid_xyz=np.isfinite(points).all(axis=1);finite_norm=np.isfinite(normals).all(axis=1);length=np.linalg.norm(np.where(finite_norm[:,None],normals,0),axis=1);valid_norm=finite_norm&(length>1e-12);valid=valid_xyz&valid_norm
    if valid.sum()<10:raise ValueError("At least 10 points with finite, nonzero normals are required")
    if neighbor_k<3:raise ValueError("neighbor_k must be >= 3")
    ids=np.flatnonzero(valid);p=points[ids].copy();n=normals[ids].copy()/length[ids,None];before=_consistency(n,p)
    out=o3d.geometry.PointCloud();out.points=o3d.utility.Vector3dVector(p);out.normals=o3d.utility.Vector3dVector(n)
    if cloud.has_colors():out.colors=o3d.utility.Vector3dVector(np.asarray(cloud.colors)[ids].copy())
    mode="normalized_only"
    if orient:
        if len(p)<neighbor_k+1:raise ValueError("neighbor_k exceeds usable crop size")
        if before is not None and before < 0.95:
            try:
                out.orient_normals_consistent_tangent_plane(neighbor_k);n=np.asarray(out.normals);mode=f"normalized_and_oriented_local_k{neighbor_k}"
            except RuntimeError:
                mode="normalized_only_orientation_failed_preserved_input_signs"
        else:
            mode="normalized_only_already_locally_consistent"
    after=_consistency(n,p)
    return PreparedCloud(out,ids,int((~valid_xyz).sum()),int((valid_xyz&~valid_norm).sum()),mode,before,after)

def estimate_spacing(cloud: o3d.geometry.PointCloud, sample_limit: int=10000, seed: int=42) -> float:
    """Mediana de distancia al vecino más cercano en muestra fija."""
    p=np.asarray(cloud.points);n=len(p)
    if n<2:raise ValueError("At least two points required")
    ids=np.arange(n) if n<=sample_limit else np.random.default_rng(seed).choice(n,sample_limit,replace=False)
    d,_=cKDTree(p).query(p[ids],k=2,workers=1);valid=d[:,1][np.isfinite(d[:,1])&(d[:,1]>0)]
    if not len(valid):raise ValueError("Could not estimate positive point spacing")
    return float(np.median(valid))

def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Conservative duplicate/degenerate cleanup, then remove unreferenced vertices."""
    mesh.remove_duplicated_vertices();mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();return mesh


def filter_supported_mesh(mesh: o3d.geometry.TriangleMesh, points: np.ndarray, normals: np.ndarray, *, max_edge: float, max_distance: float, min_normal_cosine: float = 0.25) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    """Conservatively remove unsupported faces; no face is an observed surface."""
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape or not len(p):
        raise ValueError("Se requieren puntos y normales alineados")
    if not np.isfinite(p).all() or not np.isfinite(n).all() or not np.isfinite([max_edge, max_distance, min_normal_cosine]).all() or max_edge <= 0 or max_distance <= 0 or not 0 <= min_normal_cosine <= 1:
        raise ValueError("Parámetros o datos de soporte inválidos")
    out = o3d.geometry.TriangleMesh(mesh)
    vertices, faces = np.asarray(out.vertices), np.asarray(out.triangles)
    if not len(faces):
        return clean_mesh(out), {"before": 0, "after": 0, "removed": 0}
    if not np.isfinite(vertices).all(): raise ValueError("Vértices inválidos")
    distances, nearest = cKDTree(p).query(vertices, k=1, workers=1)
    tri = vertices[faces]
    edges = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]), axis=1)
    face_normals = np.cross(edges[:, 0], -edges[:, 2])
    lengths = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(lengths[:, None], 1e-15)
    support_normals = n[nearest[faces]]
    support_lengths = np.linalg.norm(support_normals, axis=2)
    cosines = np.abs(np.einsum("ij,ikj->ik", face_normals, support_normals)) / np.maximum(support_lengths, 1e-15)
    unit_support = support_normals / np.maximum(support_lengths[:, :, None], 1e-15)
    neighbor_agreement = np.minimum.reduce((
        np.abs(np.einsum("ij,ij->i", unit_support[:, 0], unit_support[:, 1])),
        np.abs(np.einsum("ij,ij->i", unit_support[:, 1], unit_support[:, 2])),
        np.abs(np.einsum("ij,ij->i", unit_support[:, 2], unit_support[:, 0])),
    ))
    keep = ((np.max(np.linalg.norm(edges, axis=2), axis=1) <= max_edge)
            & (np.max(distances[faces], axis=1) <= max_distance)
            & (np.min(cosines, axis=1) >= min_normal_cosine)
            & (neighbor_agreement >= min_normal_cosine)
            & (lengths > 1e-15) & np.all(support_lengths > 1e-12, axis=1))
    before = len(faces)
    out.remove_triangles_by_mask((~keep).tolist())
    clean_mesh(out)
    return out, {"before": int(before), "after": int(len(out.triangles)), "removed": int(before-len(out.triangles)),
                 "max_edge": float(max_edge), "max_vertex_distance": float(max_distance),
                 "min_abs_normal_cosine": float(min_normal_cosine),
                 "interpretation": "Malla inferida entre muestras; proximidad no confirma superficie observada"}

def reconstruct_bpa(cloud: o3d.geometry.PointCloud, radii: list[float]) -> tuple[o3d.geometry.TriangleMesh,float]:
    if cloud.is_empty() or not cloud.has_normals():raise ValueError("BPA needs a nonempty cloud with normals")
    if not radii or any((not np.isfinite(r) or r<=0) for r in radii):raise ValueError("BPA radii must be positive and finite")
    start=time.perf_counter();mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud,o3d.utility.DoubleVector(sorted(set(radii))));return clean_mesh(mesh),time.perf_counter()-start

def reconstruct_poisson(cloud: o3d.geometry.PointCloud, *, depth: int=6, scale: float=1.1, density_quantile: float=.05) -> tuple[o3d.geometry.TriangleMesh,dict[str,Any],float]:
    if cloud.is_empty() or not cloud.has_normals():raise ValueError("Poisson needs a nonempty cloud with normals")
    if not 4<=depth<=9 or scale<=1 or not 0<=density_quantile<.5:raise ValueError("Unsafe/invalid Poisson depth, scale or density quantile")
    start=time.perf_counter();mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=depth,scale=scale,linear_fit=False,n_threads=1);values=np.asarray(density);threshold=float(np.quantile(values,density_quantile)) if len(values) else 0.;keep=values>=threshold
    removed=int((~keep).sum());mesh=mesh.select_by_index(np.flatnonzero(keep),cleanup=True);mesh=clean_mesh(mesh);stats={"density_quantile":density_quantile,"density_threshold":threshold,"vertices_removed_low_density":removed};return mesh,stats,time.perf_counter()-start

def mesh_quality(mesh: o3d.geometry.TriangleMesh, observed: np.ndarray, *, sample_limit: int=5000, seed: int=42) -> dict[str,Any]:
    """One-sided sampled observed-point to triangle-surface distances."""
    p=np.asarray(observed);v=len(mesh.vertices);t=len(mesh.triangles);out={"vertices":v,"triangles":t,"observed_sample":0,"distance_median":None,"distance_p95":None}
    if not v or not t or not len(p):return out
    ids=np.arange(len(p)) if len(p)<=sample_limit else np.random.default_rng(seed).choice(len(p),sample_limit,replace=False);sample=p[ids].astype(np.float32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));dist=scene.compute_distance(o3d.core.Tensor(sample)).numpy();out.update({"observed_sample":len(sample),"distance_median":float(np.median(dist)),"distance_p95":float(np.quantile(dist,.95))});return out

def vertex_source_mapping(mesh: o3d.geometry.TriangleMesh, source_points: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Nearest source index/distance per mesh vertex; approximate for Poisson."""
    v=np.asarray(mesh.vertices);p=np.asarray(source_points)
    if not len(v) or not len(p):return np.full(len(v),-1,np.int64),np.full(len(v),np.inf)
    d,i=cKDTree(p).query(v,k=1,workers=1);return i.astype(np.int64),d
