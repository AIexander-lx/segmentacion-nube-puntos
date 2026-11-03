"""Clustering espacial con presupuesto de vecinos y memoria controlados."""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def spatial_components(points: np.ndarray, eps: float=0.30, min_points: int=8, max_neighbors: int=32, min_cluster_size: int=30, max_cluster_size: int=100000) -> tuple[np.ndarray,dict,float]:
    """Componentes de puntos core kNN y asignación determinista de bordes.

    Aproxima DBSCAN: vecindarios limitados a k vecinos, componentes core y
    bordes asignados al core más cercano dentro de eps (empate: menor ID).
    No materializa listas de radio sin límite; no es DBSCAN exacto.
    """
    p=np.asarray(points,dtype=np.float64); n=len(p)
    if p.ndim!=2 or p.shape[1]!=3: raise ValueError("points debe tener forma (n,3)")
    if not np.isfinite(p).all(): raise ValueError("points debe ser finito")
    if eps<=0 or min_points<1 or max_neighbors<min_points or min_cluster_size<1 or max_cluster_size<min_cluster_size: raise ValueError("Parámetros inválidos")
    labels=np.full(n,-1,dtype=np.int32)
    if not n:return labels,{"candidate_points":0,"core_points":0,"border_points":0,"noise_points":0,"clusters":0,"oversized_neighbor_rows":0},0.0
    start=time.perf_counter(); tree=cKDTree(p); k=min(n,max_neighbors+1); distances,indices=tree.query(p,k=k,workers=1)
    if k==1: distances=distances[:,None];indices=indices[:,None]
    neighbor_counts=np.sum(distances<=eps,axis=1); core=neighbor_counts>=min_points
    src=np.repeat(np.flatnonzero(core),k); dst=indices[core].reshape(-1); ds=distances[core].reshape(-1)
    keep=(ds<=eps)&core[src]&core[dst]; src=src[keep];dst=dst[keep]
    graph=coo_matrix((np.ones(len(src),dtype=np.uint8),(src,dst)),shape=(n,n)).tocsr(); _,comp=connected_components(graph,directed=False)
    sizes=np.bincount(comp,minlength=int(comp.max())+1)
    eligible=core & (sizes[comp]>=min_cluster_size)&(sizes[comp]<=max_cluster_size)
    components=np.unique(comp[eligible]); ordered=sorted(components,key=lambda x:int(np.flatnonzero((comp==x)&eligible)[0]))
    for cid,component in enumerate(ordered): labels[(comp==component)&eligible]=cid
    # Non-core points join the closest labeled core among their capped k nearest.
    border=~core; border_rows=np.flatnonzero(border)
    if len(border_rows):
        near_ids=indices[border]; near_dist=distances[border]
        candidates=(near_dist<=eps)&(labels[near_ids]>=0)
        # Distance first, then cluster ID, then point index gives stable ties.
        row_idx, col_idx=np.nonzero(candidates)
        selected=np.full(len(border_rows),-1,dtype=np.int64)
        if len(row_idx):
            neighbor_idx=near_ids[row_idx,col_idx]
            order=np.lexsort((neighbor_idx,labels[neighbor_idx],near_dist[row_idx,col_idx],row_idx))
            first=order[np.r_[True,row_idx[order][1:]!=row_idx[order][:-1]]]
            selected[row_idx[first]]=labels[neighbor_idx[first]]
        labels[border_rows[selected>=0]]=selected[selected>=0]
    sizes_out=np.bincount(labels[labels>=0],minlength=len(ordered))
    elapsed=time.perf_counter()-start
    core_component_sizes=sizes[comp[core]]
    rejected_small=int(np.sum(core & (sizes[comp]<min_cluster_size)))
    rejected_large=int(np.sum(core & (sizes[comp]>max_cluster_size)))
    metrics={"candidate_points":int(core.sum()),"core_points":int(core.sum()),"retained_core_points":int(np.sum(labels[core]>=0)),"rejected_core_small_components":rejected_small,"rejected_core_oversized_components":rejected_large,"largest_core_component":int(core_component_sizes.max()) if len(core_component_sizes) else 0,"border_points":int(np.sum(border & (labels>=0))),"noise_points":int(np.sum(labels<0)),"clusters":len(ordered),"oversized_neighbor_rows":int(np.sum(neighbor_counts>=k)),"graph_edges_directed":int(len(src)),"max_neighbors":max_neighbors,"cluster_size_min":int(sizes_out.min()) if len(sizes_out) else 0,"cluster_size_median":float(np.median(sizes_out)) if len(sizes_out) else 0,"cluster_size_max":int(sizes_out.max()) if len(sizes_out) else 0,"method":"bounded_knn_core_components_with_nearest_core_border"}
    return labels,metrics,elapsed

def radius_components(points: np.ndarray, eps: float=.2, min_points: int=5, min_cluster_size: int=5, max_cluster_size: int=20000) -> tuple[np.ndarray,dict,float]:
    """DBSCAN-style exact radius connectivity for small, explicitly capped crops."""
    p=np.asarray(points,dtype=np.float64); n=len(p)
    if n>20000: raise ValueError("El método exacto por radio se limita a recortes <= 20,000 puntos")
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError("points debe ser XYZ finito")
    if eps<=0 or min_points<1 or min_cluster_size<1 or max_cluster_size<min_cluster_size: raise ValueError("Parámetros inválidos")
    labels=np.full(n,-1,dtype=np.int32)
    if not n:return labels,{"core_points":0,"border_points":0,"noise_points":0,"clusters":0},0.
    start=time.perf_counter();tree=cKDTree(p); neighborhoods=tree.query_ball_point(p,eps,workers=1); counts=np.fromiter((len(row) for row in neighborhoods),dtype=np.int32,count=n);core=counts>=min_points
    lengths=np.fromiter((len(row) for row in neighborhoods),dtype=np.int32,count=n); src=np.repeat(np.arange(n),lengths); dst=np.concatenate(neighborhoods).astype(np.int64,copy=False)
    keep=core[src]&core[dst]; graph=coo_matrix((np.ones(int(keep.sum()),dtype=np.uint8),(src[keep],dst[keep])),shape=(n,n)).tocsr();_,comp=connected_components(graph,directed=False);sizes=np.bincount(comp,minlength=int(comp.max())+1);eligible=core&(sizes[comp]>=min_cluster_size)&(sizes[comp]<=max_cluster_size)
    comps=np.unique(comp[eligible]);ordered=sorted(comps,key=lambda cid:int(np.flatnonzero((comp==cid)&eligible)[0]))
    for cid,component in enumerate(ordered):labels[(comp==component)&eligible]=cid
    # Border points choose closest eligible core, with stable cluster/index tie break.
    border=np.flatnonzero(~core); flat_start=np.r_[0,np.cumsum(lengths)];candidate_rows=np.repeat(np.arange(n),lengths);valid=(~core[src])&eligible[dst]
    bsrc=candidate_rows[valid];bdst=dst[valid]
    if len(bsrc):
        dist=np.linalg.norm(p[bsrc]-p[bdst],axis=1);order=np.lexsort((bdst,labels[bdst],dist,bsrc));first=order[np.r_[True,bsrc[order][1:]!=bsrc[order][:-1]]];labels[bsrc[first]]=labels[bdst[first]]
    elapsed=time.perf_counter()-start
    return labels,{"core_points":int(core.sum()),"border_points":int(np.sum((~core)&(labels>=0))),"noise_points":int(np.sum(labels<0)),"clusters":len(ordered),"neighbor_pairs":int(len(dst)),"method":"exact_radius_connected_core_components"},elapsed

def core_component_labels(points: np.ndarray, eps: float, min_points: int, max_neighbors: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Devuelve core mask, componente core y tamaños usando igual grafo kNN."""
    p=np.asarray(points,dtype=np.float64);n=len(p)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError("points debe ser XYZ finito")
    if eps<=0 or min_points<1 or max_neighbors<min_points:raise ValueError("Parámetros inválidos")
    if not n:return np.zeros(0,bool),np.zeros(0,np.int32),np.zeros(0,np.int64)
    tree=cKDTree(p);k=min(n,max_neighbors+1);dist,idx=tree.query(p,k=k,workers=1)
    if k==1:dist=dist[:,None];idx=idx[:,None]
    core=np.sum(dist<=eps,axis=1)>=min_points;src=np.repeat(np.flatnonzero(core),k);dst=idx[core].reshape(-1);d=dist[core].reshape(-1);keep=(d<=eps)&core[src]&core[dst]
    graph=coo_matrix((np.ones(int(keep.sum()),np.uint8),(src[keep],dst[keep])),shape=(n,n)).tocsr();_,components=connected_components(graph,directed=False)
    sizes=np.bincount(components,minlength=int(components.max())+1);return core,components.astype(np.int32),sizes
