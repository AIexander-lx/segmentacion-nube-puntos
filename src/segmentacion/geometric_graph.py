"""Grafo experimental de afinidad geométrica para regiones espaciales acotadas."""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def geometric_components(points: np.ndarray, normals: np.ndarray, *, eps: float=.15, max_neighbors: int=24, normal_angle_degrees: float=35., colors: np.ndarray|None=None, color_distance: float|None=None, curvature: np.ndarray|None=None, curvature_tolerance: float|None=None, min_component_size: int=5, max_edges: int=2_000_000, allow_invalid_normals: bool=True) -> tuple[np.ndarray,dict,float]:
    """Conecta vecinos cercanos compatibles; normal con abs(dot) admite signo invertido.

    PCA/color son filtros opcionales. El grafo kNN está acotado y falla antes
    de construirlo si el presupuesto máximo de aristas se excede.
    """
    p=np.asarray(points,dtype=float);n=len(p);norm=np.asarray(normals,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError("points debe ser XYZ finito")
    if norm.shape!=p.shape:raise ValueError("normals debe tener forma (n,3)")
    if eps<=0 or max_neighbors<1 or not 0<=normal_angle_degrees<=90 or min_component_size<1 or max_edges<1:raise ValueError("Parámetros geométricos inválidos")
    if colors is not None:
        colors=np.asarray(colors,dtype=float)
        if colors.shape!=p.shape or not np.isfinite(colors).all():raise ValueError("colors debe ser RGB finito con forma (n,3)")
        if color_distance is not None and color_distance<0:raise ValueError("color_distance debe ser >= 0")
    if curvature is not None:
        curvature=np.asarray(curvature,dtype=float).reshape(-1)
        if len(curvature)!=n:raise ValueError("curvature debe tener n valores")
        if curvature_tolerance is not None and curvature_tolerance<0:raise ValueError("curvature_tolerance debe ser >= 0")
    if not n:return np.full(0,-1,np.int32),{"components":0,"accepted_edges":0,"rejected_edges":0},0.
    start=time.perf_counter();k=min(n,max_neighbors+1)
    if n*(k-1)>max_edges:raise MemoryError(f"El presupuesto superior del grafo excede max_edges={max_edges}")
    dist,idx=cKDTree(p).query(p,k=k,workers=1)
    if k==1:dist=dist[:,None];idx=idx[:,None]
    src=np.repeat(np.arange(n),k);dst=idx.reshape(-1);d=dist.reshape(-1);near=(src!=dst)&(d<=eps);src=src[near];dst=dst[near];d=d[near];candidate_count=len(src)
    if candidate_count>max_edges:raise MemoryError(f"Grafo excede max_edges={max_edges}; reduzca recorte o max_neighbors")
    lengths=np.linalg.norm(norm,axis=1);valid=np.isfinite(norm).all(axis=1)&(lengths>1e-12);unit=np.zeros_like(norm);unit[valid]=norm[valid]/lengths[valid,None]
    dot=np.abs(np.einsum("ij,ij->i",unit[src],unit[dst])); compatible=(dot>=np.cos(np.deg2rad(normal_angle_degrees)))
    normal_ok=(~(valid[src]&valid[dst]) | compatible) if allow_invalid_normals else (valid[src]&valid[dst]&compatible)
    ok=normal_ok
    if colors is not None and color_distance is not None:ok &= np.linalg.norm(colors[src]-colors[dst],axis=1)<=color_distance
    if curvature is not None and curvature_tolerance is not None:ok &= np.abs(curvature[src]-curvature[dst])<=curvature_tolerance
    a=src[ok];b=dst[ok];graph=coo_matrix((np.ones(len(a),dtype=np.uint8),(a,b)),shape=(n,n)).tocsr();_,component=connected_components(graph,directed=False);sizes=np.bincount(component,minlength=int(component.max())+1);keep=sizes[component]>=min_component_size
    components=np.unique(component[keep]);ordered=sorted(components,key=lambda cid:int(np.flatnonzero((component==cid)&keep)[0]));labels=np.full(n,-1,np.int32)
    for label,cid in enumerate(ordered):labels[(component==cid)&keep]=label
    elapsed=time.perf_counter()-start
    return labels,{"components":len(ordered),"noise_points":int(np.sum(labels<0)),"accepted_edges_directed":int(ok.sum()),"rejected_edges_directed":int(candidate_count-ok.sum()),"normal_rejected_edges":int((~normal_ok).sum()),"candidate_edges_directed":candidate_count,"max_neighbors":max_neighbors,"max_edges":max_edges,"component_size_min":int(min(sizes[component[keep]])) if np.any(keep) else 0,"component_size_max":int(max(sizes[component[keep]])) if np.any(keep) else 0},elapsed
