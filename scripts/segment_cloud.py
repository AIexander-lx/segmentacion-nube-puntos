"""Segmentación geométrica inicial: planos RANSAC y componentes espaciales."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np, open3d as o3d
from segmentacion.io import load_ply
from segmentacion.planes import extract_planes
from segmentacion.clustering import spatial_components
from segmentacion.segmentation import run_segmentation

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--original",type=Path,default=ROOT/"FARO_Scan_Full_Sub1_down3_with_normals.ply");p.add_argument("--global-cloud",type=Path,default=ROOT/"outputs/processed/global.ply");p.add_argument("--mapping",type=Path,default=ROOT/"outputs/processed/global_mapping.npz");p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/segmented/run_001");p.add_argument("--seed",type=int,default=42);p.add_argument("--distance-threshold",type=float,default=.05);p.add_argument("--iterations",type=int,default=500);p.add_argument("--min-plane-points",type=int,default=3000);p.add_argument("--max-planes",type=int,default=6);p.add_argument("--vertical-axis",choices=("x","y","z"),default=None);p.add_argument("--eps",type=float,default=.30);p.add_argument("--min-points",type=int,default=8);p.add_argument("--max-neighbors",type=int,default=32);p.add_argument("--min-cluster-size",type=int,default=30);p.add_argument("--max-cluster-size",type=int,default=100000);p.add_argument("--sample-size",type=int,default=0,help="Muestra reproducible; modo diagnóstico sin exportar PLY original");a=p.parse_args()
 try:
  if a.sample_size:
   cloud=load_ply(a.global_cloud); pts=np.asarray(cloud.points); rng=np.random.default_rng(a.seed); ids=np.sort(rng.choice(len(pts),min(a.sample_size,len(pts)),replace=False)); sub=pts[ids]
   pl,models,tr=extract_planes(sub,a.distance_threshold,a.iterations,a.min_plane_points,a.max_planes,a.seed,vertical_axis={None:None,"x":0,"y":1,"z":2}[a.vertical_axis]); rest=np.flatnonzero(pl<0); cl,cm,tc=spatial_components(sub[rest],a.eps,a.min_points,a.max_neighbors,a.min_cluster_size,a.max_cluster_size)
   report={"mode":"sample_diagnostic_only","sample_size":len(sub),"seed":a.seed,"planes":models,"plane_count":len(models),"clusters":cm["clusters"],"unassigned_sample":int(np.sum(pl<0)-np.sum(cl>=0)),"timings_seconds":{"ransac":tr,"clustering":tc},"outputs":{}};a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2));return 0
  report=run_segmentation(a.original,a.global_cloud,a.mapping,a.output_dir,seed=a.seed,distance_threshold=a.distance_threshold,ransac_iterations=a.iterations,min_plane_points=a.min_plane_points,max_planes=a.max_planes,vertical_axis={None:None,"x":0,"y":1,"z":2}[a.vertical_axis],eps=a.eps,min_points=a.min_points,max_neighbors=a.max_neighbors,min_cluster_size=a.min_cluster_size,max_cluster_size=a.max_cluster_size)
  # Vista XY/XZ/YZ del PLY coloreado, muestreo determinista y no GUI.
  colored=load_ply(report["outputs"]["ply"]); xyz=np.asarray(colored.points); rgb=np.asarray(colored.colors); ids=np.random.default_rng(a.seed).choice(len(xyz),min(100000,len(xyz)),replace=False)
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  fig,axs=plt.subplots(1,3,figsize=(15,5))
  for ax,(i,j,title) in zip(axs,((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))):ax.scatter(xyz[ids,i],xyz[ids,j],c=rgb[ids],s=.15,linewidths=0,rasterized=True);ax.set_title(title);ax.set_xlabel("XYZ"[i]);ax.set_ylabel("XYZ"[j]);ax.set_aspect("equal",adjustable="datalim")
  fig.tight_layout();fig.savefig(a.output_dir/"segmentation_projections.png",dpi=180);plt.close(fig);report["outputs"]["figure"]=str(a.output_dir/"segmentation_projections.png");(a.output_dir/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2));return 0
 except (OSError,ValueError) as e:p.error(str(e));return 2
if __name__=="__main__":raise SystemExit(main())
