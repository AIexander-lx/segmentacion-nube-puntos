"""Orquesta planos, componentes espaciales y propagación al PLY original."""
from __future__ import annotations
import json, time, sys, ctypes
from pathlib import Path
from typing import Any
import numpy as np
import open3d as o3d
from .clustering import spatial_components
from .io import load_ply
from .planes import extract_planes
from .preprocessing import propagate_voxel_labels

def _peak_working_set_bytes() -> int | None:
    """Windows peak resident working set; None when OS API is unavailable."""
    if sys.platform != "win32": return None
    class Counters(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    try:
        handle=ctypes.windll.kernel32.GetCurrentProcess()
        ok=ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except (AttributeError,OSError): return None

def run_segmentation(original_path: Path, global_path: Path, mapping_path: Path, output_dir: Path, *, seed: int=42, distance_threshold: float=.05, ransac_iterations: int=500, min_plane_points: int=3000, max_planes: int=6, vertical_axis: int|None=None, eps: float=.30, min_points: int=8, max_neighbors: int=32, min_cluster_size: int=30, max_cluster_size: int=100000) -> dict[str,Any]:
    """Ejecuta una corrida nueva; falla si output_dir ya existe."""
    output_dir=output_dir.resolve()
    if output_dir.exists(): raise FileExistsError(f"La salida ya existe; elige una carpeta nueva: {output_dir}")
    load_start=time.perf_counter(); original=load_ply(original_path); reduced=load_ply(global_path); maps=np.load(mapping_path); mapping=maps["original_to_voxel"]; load_seconds=time.perf_counter()-load_start
    if len(mapping)!=len(original.points): raise ValueError("El mapeo no coincide con la nube original")
    valid=mapping>=0
    if len(np.unique(mapping[valid]))!=len(reduced.points) or (mapping[valid].min(initial=0)<0) or (mapping[valid].max(initial=-1)>=len(reduced.points)): raise ValueError("Índices del mapeo inválidos o incompletos")
    counts=np.bincount(mapping[valid],minlength=len(reduced.points))
    if len(counts)!=len(reduced.points) or np.any(counts==0): raise ValueError("Hay vóxeles sin correspondencia en el mapa")
    points=np.asarray(reduced.points)
    plane_labels,planes,ransac_time=extract_planes(points,distance_threshold,ransac_iterations,min_plane_points,max_planes,seed,vertical_axis=vertical_axis)
    rest=np.flatnonzero(plane_labels<0); cluster_local,cluster_metrics,cluster_time=spatial_components(points[rest],eps,min_points,max_neighbors,min_cluster_size,max_cluster_size)
    labels=plane_labels.copy(); cluster_mask=cluster_local>=0; labels[rest[cluster_mask]]=len(planes)+cluster_local[cluster_mask]
    propagate_start=time.perf_counter(); original_labels=propagate_voxel_labels(labels,mapping); propagation_time=time.perf_counter()-propagate_start
    # Paleta determinista HSV, gris para ruido/sin asignar.
    k=int(labels.max()+1) if np.any(labels>=0) else 0; palette=np.zeros((k,3),dtype=float)
    for i in range(k): palette[i]=__import__("colorsys").hsv_to_rgb((i*0.61803398875)%1,.78,.95)
    colors=np.full((len(original_labels),3),.35); assigned=original_labels>=0
    if k: colors[assigned]=palette[original_labels[assigned]]
    colored=o3d.geometry.PointCloud(original); colored.colors=o3d.utility.Vector3dVector(colors)
    output_dir.mkdir(parents=True); ply_path=output_dir/"segmented_original.ply"
    export_start=time.perf_counter()
    if not o3d.io.write_point_cloud(str(ply_path),colored): raise OSError(f"Falló exportación {ply_path}")
    export_time=time.perf_counter()-export_start
    np.savez_compressed(output_dir/"labels.npz",global_labels=labels,original_labels=original_labels,colors=palette)
    unique,segment_counts=np.unique(labels[labels>=0],return_counts=True)
    report={"input":{"original":str(original_path.resolve()),"global":str(global_path.resolve()),"mapping":str(mapping_path.resolve())},"parameters":{"seed":seed,"distance_threshold":distance_threshold,"ransac_iterations":ransac_iterations,"min_plane_points":min_plane_points,"max_planes":max_planes,"vertical_axis":vertical_axis,"eps":eps,"min_points":min_points,"max_neighbors":max_neighbors,"min_cluster_size":min_cluster_size,"max_cluster_size":max_cluster_size},"versions":{"python":sys.version.split()[0],"numpy":np.__version__,"open3d":o3d.__version__},"counts":{"original_points":len(original.points),"global_points":len(points),"planes":len(planes),"plane_points_global":int(np.sum(plane_labels>=0)),"cluster_points_global":int(np.sum(cluster_local>=0)),"noise_points_global":int(np.sum(labels<0)),"assigned_global":int(np.sum(labels>=0)),"global_coverage_percent":100*float(np.mean(labels>=0)),"clusters":cluster_metrics["clusters"],"assigned_original":int(assigned.sum()),"unassigned_original":int((~assigned).sum()),"original_coverage_percent":100*float(assigned.sum()/len(original.points)),"segments":{str(int(i)):int(c) for i,c in zip(unique,segment_counts)}},"planes":planes,"clustering":cluster_metrics,"timings_seconds":{"load":load_seconds,"ransac":ransac_time,"clustering":cluster_time,"propagation":propagation_time,"export":export_time},"peak_working_set_bytes":_peak_working_set_bytes(),"outputs":{"ply":str(ply_path),"labels":str(output_dir/"labels.npz"),"ply_bytes":ply_path.stat().st_size}}
    (output_dir/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report
