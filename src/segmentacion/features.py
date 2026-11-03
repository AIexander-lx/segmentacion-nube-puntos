"""Descriptores de forma locales mediante PCA de k vecinos acotados."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class LocalFeatures:
    values: np.ndarray  # columnas: linearity, planarity, scattering, curvature
    valid: np.ndarray
    neighbor_count: np.ndarray

def local_pca_features(points: np.ndarray, *, k: int=20, batch_size: int=4096) -> LocalFeatures:
    """Calcula PCA local en lotes; inválidos/degenerados quedan en NaN y valid=False."""
    p=np.asarray(points,dtype=np.float64);n=len(p)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError("points debe ser XYZ finito")
    if k<3 or batch_size<1:raise ValueError("k >= 3 y batch_size > 0 requeridos")
    values=np.full((n,4),np.nan);valid=np.zeros(n,bool);counts=np.zeros(n,np.int16)
    if n<3:return LocalFeatures(values,valid,counts)
    tree=cKDTree(p);kk=min(k,n)
    for start in range(0,n,batch_size):
        end=min(start+batch_size,n);_,idx=tree.query(p[start:end],k=kk,workers=1)
        if kk==1:idx=idx[:,None]
        counts[start:end]=kk
        neighborhoods=p[idx];centered=neighborhoods-neighborhoods.mean(axis=1,keepdims=True)
        cov=np.einsum("bki,bkj->bij",centered,centered)/kk
        eig=np.linalg.eigvalsh(cov)[:,::-1];eig=np.maximum(eig,0);total=eig.sum(axis=1);good=np.isfinite(eig).all(axis=1)&(total>1e-15)&(eig[:,0]>1e-15)
        l1,l2,l3=eig.T
        features=np.column_stack(((l1-l2)/np.maximum(l1,1e-15),(l2-l3)/np.maximum(l1,1e-15),l3/np.maximum(l1,1e-15),l3/np.maximum(total,1e-15)))
        values[start:end][good]=features[good];valid[start:end]=good
    return LocalFeatures(values,valid,counts)
