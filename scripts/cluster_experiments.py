"""Compara clustering en recortes espaciales contiguos de la nube global."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np,open3d as o3d
from segmentacion.clustering import spatial_components,radius_components

def main():
 out=ROOT/"outputs/experiments/cluster_diagnostic_v4";out.mkdir(parents=True,exist_ok=False)
 cloud=o3d.io.read_point_cloud(str(ROOT/"outputs/processed/global.ply"));p=np.asarray(cloud.points);baseline=np.load(ROOT/"outputs/segmented/global_seed42/labels.npz")["global_labels"]
 cell_size=2.5;keys=np.floor((p-p.min(axis=0))/cell_size).astype(np.int32);unique,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True);plane_fraction=np.bincount(inv,weights=((baseline>=0)&(baseline<6)),minlength=len(unique))/counts
 eligible=np.flatnonzero((counts>=250)&(counts<=12000));selected={}
 if len(eligible):
  selected["dense"] = int(eligible[np.argmax(counts[eligible])])
  selected["sparse"] = int(eligible[np.argmin(counts[eligible])])
  selected["near_planes"] = int(eligible[np.argmin(np.abs(plane_fraction[eligible]-.5))])
  object_cells=eligible[plane_fraction[eligible]<.05];selected["nonplane_crop"] = int(object_cells[np.argmin(counts[object_cells])] if len(object_cells) else eligible[np.argsort(counts[eligible])[min(1,len(eligible)-1)]])
  if len(object_cells)>1:
   order=np.argsort(counts[object_cells]); selected["small_nonplane_candidate"]=int(object_cells[order[min(1,len(order)-1)]])
 # De-duplicate crops; spatially adjacent alternatives maintain locality.
 used=set();regions={}
 for name,cell in selected.items():
  if cell in used: continue
  used.add(cell);ids=np.flatnonzero(inv==cell);regions[name]={"indices":ids,"point_count":len(ids),"cell_key":unique[cell].tolist(),"plane_fraction":float(plane_fraction[cell])}
 configs=[(.15,5,32),(.25,8,32),(.25,8,64),(.35,8,64)]
 report={"cell_size":cell_size,"selection":"contiguous axis-aligned grid cells, 250–12,000 global points","regions":{},"experiments":[]}
 for name,r in regions.items():
  ids=r["indices"];sub=p[ids];report["regions"][name]={k:v for k,v in r.items() if k!="indices"}
  for eps,minpts,k in configs:
   labels,metrics,elapsed=spatial_components(sub,eps,minpts,k,10,20000)
   report["experiments"].append({"region":name,"method":"bounded_knn_with_border","eps":eps,"min_points":minpts,"max_neighbors":k,"elapsed_seconds":elapsed,"assigned":int(np.sum(labels>=0)),"noise":int(np.sum(labels<0)),"clusters":metrics["clusters"],"core":metrics["core_points"],"border":metrics["border_points"],"saturation":metrics["oversized_neighbor_rows"],"size_min":metrics["cluster_size_min"],"size_median":metrics["cluster_size_median"],"size_max":metrics["cluster_size_max"]})
  if len(sub)<=5000:
   radius_labels,radius_metrics,radius_time=radius_components(sub,.25,8,10,20000)
   report["experiments"].append({"region":name,"method":"exact_radius_search","eps":.25,"min_points":8,"elapsed_seconds":radius_time,"assigned":int(np.sum(radius_labels>=0)),"noise":int(np.sum(radius_labels<0)),"clusters":radius_metrics["clusters"],"core":radius_metrics["core_points"],"border":radius_metrics["border_points"]})
   ref=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub));t=time.perf_counter();ref_labels=np.asarray(ref.cluster_dbscan(eps=.25,min_points=8,print_progress=False));elapsed=time.perf_counter()-t
   report["experiments"].append({"region":name,"method":"open3d_dbscan_reference","eps":.25,"min_points":8,"elapsed_seconds":elapsed,"assigned":int(np.sum(ref_labels>=0)),"noise":int(np.sum(ref_labels<0)),"clusters":int(ref_labels.max()+1) if np.any(ref_labels>=0) else 0})
 # Save compact spatial diagnostic figure, color noise grey and baseline plane/cluster IDs deterministically.
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,max(1,len(regions)),figsize=(5*max(1,len(regions)),5));axs=np.atleast_1d(axs)
 for ax,(name,r) in zip(axs,regions.items()):
  ids=r["indices"];lab=baseline[ids];colors=np.full((len(ids),3),.45);ok=lab>=0;colors[ok]=plt.cm.tab20(lab[ok]%20)[:,:3];ax.scatter(p[ids,0],p[ids,1],c=colors,s=.4,linewidths=0);ax.set_title(f"{name}: n={len(ids)}; gris=ruido baseline");ax.set_xlabel("X");ax.set_ylabel("Y");ax.set_aspect("equal",adjustable="datalim")
 fig.tight_layout();fig.savefig(out/"spatial_crops_baseline.png",dpi=180);plt.close(fig)
 report["figure"]=str(out/"spatial_crops_baseline.png");(out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
