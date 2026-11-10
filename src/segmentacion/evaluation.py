"""Métricas internas y evaluación de instancias solo con anotación manual."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def internal_summary(labels: np.ndarray) -> dict:
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer):raise ValueError("labels debe ser vector entero")
    ids,counts=np.unique(labels[labels>=0],return_counts=True)
    return {"point_count":len(labels),"assigned":int(np.sum(labels>=0)),"unassigned":int(np.sum(labels<0)),"coverage_percent":100*float(np.mean(labels>=0)) if len(labels) else 0.,"instance_count":len(ids),"sizes":{"min":int(counts.min()) if len(counts) else 0,"median":float(np.median(counts)) if len(counts) else 0.,"max":int(counts.max()) if len(counts) else 0},"sizes_by_id":{str(int(i)):int(c) for i,c in zip(ids,counts)}}

def compare_runs(first: np.ndarray, second: np.ndarray) -> dict:
    """Compara cobertura/tamaños sin asumir equivalencia de IDs entre corridas."""
    a=np.asarray(first);b=np.asarray(second)
    if a.shape!=b.shape:raise ValueError("Corridas deben compartir cantidad y orden de Índices")
    return {"index_space_count":len(a),"first":internal_summary(a),"second":internal_summary(b),"both_assigned":int(np.sum((a>=0)&(b>=0))),"only_first_assigned":int(np.sum((a>=0)&(b<0))),"only_second_assigned":int(np.sum((a<0)&(b>=0))),"ids_compared_as_equivalent":False}

def create_annotation(path: Path, original_indices: np.ndarray, *, region: str, categories: list[str], provenance: str="human annotation pending") -> Path:
    ids=np.asarray(original_indices,dtype=np.int64)
    if ids.ndim!=1 or np.any(ids<0):raise ValueError("original_indices debe ser vector de IDs no negativos")
    payload={"schema":"manual-point-annotations-v1","created_date":date.today().isoformat(),"region":region,"index_space":"original PLY row indices","categories_available":categories,"provenance":provenance,"warning":"Empty labels are unlabeled; predictions are not ground truth.","points":[{"original_index":int(i),"category":None,"instance_id":None} for i in ids]}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8");return path

def read_annotation(path: Path) -> dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema")!="manual-point-annotations-v1" or data.get("index_space")!="original PLY row indices":raise ValueError("Formato de anotación no compatible")
    ids=[r["original_index"] for r in data["points"]]
    if len(ids)!=len(set(ids)) or any(i<0 for i in ids):raise ValueError("Índices anotados duplicados o inválidos")
    return data

def evaluate_instances(annotation: dict, predicted_labels: np.ndarray) -> dict:
    """IoU con asignación húngara sobre filas manualmente etiquetadas."""
    rows=annotation["points"];pred=np.asarray(predicted_labels)
    if pred.ndim!=1 or (rows and max(r["original_index"] for r in rows)>=len(pred)):raise ValueError("Predicción no está alineada al PLY original")
    valid=[r for r in rows if r.get("instance_id") is not None and pred[r["original_index"]]>=0]
    gt_ids=sorted({str(r["instance_id"]) for r in valid});pr_ids=np.unique([pred[r["original_index"]] for r in valid]).astype(int)
    if not gt_ids or not len(pr_ids):return {"annotated_assigned_points":len(valid),"matched_instances":[],"mean_iou":None,"fragmentation_by_manual_instance":{}}
    matrix=np.zeros((len(gt_ids),len(pr_ids)),dtype=np.int64);gmap={x:i for i,x in enumerate(gt_ids)};pmap={int(x):i for i,x in enumerate(pr_ids)}
    gt_by_index={r["original_index"]:str(r["instance_id"]) for r in valid}
    for r in valid:matrix[gmap[str(r["instance_id"])],pmap[int(pred[r["original_index"]])]]+=1
    gt_count=np.bincount([gmap[str(r["instance_id"])] for r in rows if r.get("instance_id") is not None],minlength=len(gt_ids));pred_rows=[r for r in rows if r.get("instance_id") is not None and pred[r["original_index"]]>=0];pr_count=np.bincount([pmap[int(pred[r["original_index"]])] for r in pred_rows],minlength=len(pr_ids));union=gt_count[:,None]+pr_count[None,:]-matrix;iou=matrix/np.maximum(union,1);ri,ci=linear_sum_assignment(-iou);matches=[{"manual_instance":gt_ids[i],"predicted_instance":int(pr_ids[j]),"iou":float(iou[i,j]),"intersection":int(matrix[i,j])} for i,j in zip(ri,ci)]
    fragments={g:len({int(pred[r["original_index"]]) for r in valid if str(r["instance_id"])==g}) for g in gt_ids}
    return {"annotated_assigned_points":len(valid),"matched_instances":matches,"mean_iou":float(np.mean([m["iou"] for m in matches])) if matches else None,"fragmentation_by_manual_instance":fragments,"note":"IoU is reported only for manually annotated points; no semantic accuracy is inferred."}

def semantic_annotation_summary(annotation: dict) -> dict:
    """Interfaz futura: cuenta manuales semánticos sin calcular accuracy."""
    labeled=[r for r in annotation["points"] if r.get("category") is not None]
    return {"manually_labeled_points":len(labeled),"categories":sorted({str(r["category"]) for r in labeled}),"accuracy_computed":False}
