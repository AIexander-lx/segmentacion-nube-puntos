"""Conversión de etiquetas a colores y utilidades para recortes por coordenadas."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import open3d as o3d
from .io import load_ply

def deterministic_palette(count: int) -> np.ndarray:
    """Paleta estable HSV para IDs; ruido (-1) se representa en gris."""
    import colorsys
    if count<0: raise ValueError("count debe ser >= 0")
    return np.asarray([colorsys.hsv_to_rgb((i*.61803398875)%1,.78,.95) for i in range(count)],dtype=float).reshape((-1,3))

def label_colors(labels: np.ndarray, palette: np.ndarray|None=None) -> np.ndarray:
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer) or np.any(labels < -1):raise ValueError("labels debe ser un vector entero >= -1")
    k=int(labels.max()+1) if np.any(labels>=0) else 0;colors=np.full((len(labels),3),.45)
    if k:
        if palette is None:palette=deterministic_palette(k)
        palette=np.asarray(palette,dtype=float)
        if palette.shape!=(k,3):raise ValueError("La paleta no coincide con el máximo ID")
        ok=labels>=0;colors[ok]=palette[labels[ok]]
    return colors

def normals_colors(normals: np.ndarray) -> np.ndarray:
    """Muestra componentes normales [-1,1] como RGB [0,1]; inválidos en gris."""
    n=np.asarray(normals,dtype=float);out=np.full(n.shape,.45)
    valid=np.isfinite(n).all(axis=1)&(np.linalg.norm(n,axis=1)>1e-12)
    unit=n[valid]/np.linalg.norm(n[valid],axis=1,keepdims=True);out[valid]=(unit+1)/2
    return out

def roi_indices(points: np.ndarray, bounds: tuple[float,float,float,float,float,float]) -> np.ndarray:
    """Índices que caen dentro de xmin,xmax,ymin,ymax,zmin,zmax inclusivos."""
    p=np.asarray(points,dtype=float);b=np.asarray(bounds,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or b.shape!=(6,) or not np.isfinite(b).all():raise ValueError("Se requieren puntos XYZ y seis límites finitos")
    lo=b[[0,2,4]];hi=b[[1,3,5]]
    if np.any(lo>hi):raise ValueError("Cada mínimo debe ser <= su máximo")
    return np.flatnonzero(np.isfinite(p).all(axis=1)&np.all((p>=lo)&(p<=hi),axis=1))

def labels_for_resolution(label_file: Path, resolution: str, original_count: int, resolution_count: int, mapping_file: Path|None=None) -> np.ndarray:
    """Alinea etiquetas guardadas a original/global/detail mediante mapa NPZ."""
    with np.load(label_file) as data: keys=set(data.files); original=data["original_labels"] if "original_labels" in keys else None; global_labels=data["global_labels"] if "global_labels" in keys else None
    if resolution=="original":
        if original is not None and len(original)==original_count:return original.copy()
        if global_labels is not None and mapping_file is not None:
            m=np.load(mapping_file)["original_to_voxel"]
            if len(m)==original_count:return np.where(m>=0,global_labels[np.maximum(m,0)],-1).astype(np.int32)
    elif resolution in ("global","detail"):
        if resolution=="global" and global_labels is not None and len(global_labels)==resolution_count:return global_labels.copy()
        if original is not None and mapping_file is not None:
            with np.load(mapping_file) as mdata: reps=mdata["voxel_to_representative"]
            if len(reps)==resolution_count and np.all((reps>=0)&(reps<len(original))):return original[reps].copy()
    raise ValueError(f"Etiquetas incompatibles con resolución {resolution}")

def save_roi(points: np.ndarray, selected: np.ndarray, output: Path, *, resolution: str, representative_original: np.ndarray|None=None, original_to_voxel: np.ndarray|None=None) -> Path:
    """Guarda Índices y correspondencias, además de los XYZ seleccionados."""
    ids=np.asarray(selected,dtype=np.int64);p=np.asarray(points)
    if np.any(ids<0)|np.any(ids>=len(p)):raise ValueError("índice ROI fuera de rango")
    fields={"resolution_indices":ids,"points_xyz":p[ids],"resolution":np.asarray([resolution])}
    if representative_original is not None:fields["representative_original_indices"]=np.asarray(representative_original)[ids]
    if original_to_voxel is not None:
        m=np.asarray(original_to_voxel)
        if np.any(m>=len(p)):raise ValueError("original_to_voxel contiene índice fuera de la resolución")
        voxels=np.zeros(len(p),bool);voxels[ids]=True
        fields["original_indices"]=np.flatnonzero((m>=0)&voxels[np.maximum(m,0)])
        fields["original_to_voxel"]=m[fields["original_indices"]]
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**fields);return output
