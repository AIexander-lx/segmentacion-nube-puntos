"""Recupera y caracteriza el componente core gigante de la nube global."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from segmentacion.clustering import core_component_labels,spatial_components
from segmentacion.planes import extract_planes
from segmentacion.features import local_pca_features
from segmentacion.geometric_graph import geometric_components

def main():
 out=ROOT/"outputs/experiments/giant_component_v3";out.mkdir(parents=True,exist_ok=False)
 global_cloud=o3d.io.read_point_cloud(str(ROOT/"outputs/processed/global.ply"));pts=np.asarray(global_cloud.points);normals=np.asarray(global_cloud.normals);colors=np.asarray(global_cloud.colors)
 pl,models,_=extract_planes(pts,.05,500,3000,6,42,vertical_axis=None);rest=np.flatnonzero(pl<0);core,component,sizes=core_component_labels(pts[rest],.25,8,64);giant_component=int(np.argmax(sizes));mask=core&(component==giant_component);ids=rest[mask];p=pts[ids];nrm=normals[ids];rgb=colors[ids]
 np.savez_compressed(out/"giant_indices.npz",global_indices=ids,component_id=np.array([giant_component]),plane_labels=pl[ids])
 # PCA is batched; local spacing uses a fixed deterministic sample.
 t=time.perf_counter();feat=local_pca_features(p,k=20,batch_size=2048);pca_time=time.perf_counter()-t
 rng=np.random.default_rng(42);sample=p[rng.choice(len(p),min(20000,len(p)),replace=False)];dt,_=cKDTree(p).query(sample,k=9,workers=1);nn=dt[:,1]
 report={"global_points":len(pts),"planes_excluded":len(rest),"core_component_count":int(len(sizes)),"giant_component_id":giant_component,"giant_points":len(ids),"giant_fraction_of_global":len(ids)/len(pts),"xyz":{"min":p.min(axis=0).tolist(),"max":p.max(axis=0).tolist(),"quantiles":np.quantile(p,[.01,.25,.5,.75,.99],axis=0).tolist()},"rgb":{"min":rgb.min(axis=0).tolist(),"max":rgb.max(axis=0).tolist(),"mean":rgb.mean(axis=0).tolist(),"std":rgb.std(axis=0).tolist()},"normals":{"finite":int(np.isfinite(nrm).all(axis=1).sum()),"zero":int((np.linalg.norm(nrm,axis=1)<1e-12).sum()),"mean":np.nanmean(nrm,axis=0).tolist(),"axis_abs_fraction_ge_0_9":(np.mean(np.abs(nrm)>=.9,axis=0)).tolist()},"local_density_sample":{"n":len(sample),"k9_distance_quantiles":np.quantile(nn,[.1,.5,.9,.99]).tolist()},"pca":{"seconds":pca_time,"valid":int(feat.valid.sum()),"invalid":int((~feat.valid).sum()),"feature_medians":np.nanmedian(feat.values,axis=0).tolist(),"columns":["linearity","planarity","scattering","curvature"]},"spatial_cells":{}}
 # Contiguous 0.75-unit cells from the giant region; choose dense, sparse and normal-heterogeneous candidates.
 cell=1.5;keys=np.floor((p-p.min(axis=0))/cell).astype(np.int32);uk,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True);eligible=np.flatnonzero((counts>=60)&(counts<=3000));selected={}
 if len(eligible):
  selected["dense"]=int(eligible[np.argmax(counts[eligible])]);selected["sparse"]=int(eligible[np.argmin(counts[eligible])])
  lens=np.linalg.norm(nrm,axis=1);valid=lens>1e-8;unit=np.zeros_like(nrm);unit[valid]=nrm[valid]/lens[valid,None]
  valid_count=np.bincount(inv[valid],minlength=len(uk));means=np.column_stack([np.bincount(inv[valid],weights=unit[valid,j],minlength=len(uk)) for j in range(3)])
  means/=np.maximum(valid_count[:,None],1);variation=1-np.linalg.norm(means,axis=1)
  selected["normal_transition"]=int(eligible[np.argmax(variation[eligible])])
 # Save before visualization: giant colored red, sampled non-giant context grey.
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,3,figsize=(15,5));pairs=((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))
 for ax,(a,b,title) in zip(axs,pairs):ax.scatter(p[:,a],p[:,b],c=rgb,s=.12,linewidths=0,rasterized=True);ax.set_title(f"Componente gigante ? {title}");ax.set_xlabel("XYZ"[a]);ax.set_ylabel("XYZ"[b]);ax.set_aspect("equal",adjustable="datalim")
 fig.tight_layout();fig.savefig(out/"giant_before.png",dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,7));other=np.flatnonzero(pl<0);other=other[~np.isin(other,ids)];ctx=other[rng.choice(len(other),min(60000,len(other)),replace=False)];ax.scatter(pts[ctx,0],pts[ctx,1],c="lightgray",s=.1);ax.scatter(p[:,0],p[:,1],c="red",s=.12);ax.set_title("Componente gigante (rojo), contexto residual (gris)");fig.tight_layout();fig.savefig(out/"giant_context.png",dpi=180);plt.close(fig)
 for name,c in selected.items():
  ii=np.flatnonzero(inv==c);sub=p[ii];subn=nrm[ii];subrgb=rgb[ii];subcurv=feat.values[ids*0+ii,3] if False else feat.values[np.flatnonzero(mask)[ii],3]
  row={"points":len(ii),"cell_key":uk[c].tolist(),"normal_variation":float(variation[c]),"methods":[]}
  base,bm,bt=spatial_components(sub,.25,8,64,10,20000);row["methods"].append({"method":"baseline_spatial","clusters":bm["clusters"],"assigned":int((base>=0).sum()),"noise":int((base<0).sum()),"seconds":bt})
  for angle in (20.,35.,55.):
   gl,gm,gt=geometric_components(sub,subn,eps=.12,max_neighbors=24,normal_angle_degrees=angle,min_component_size=10,max_edges=500000)
   row["methods"].append({"method":"normal_graph","normal_angle":angle,"clusters":gm["components"],"assigned":int((gl>=0).sum()),"noise":gm["noise_points"],"accepted_edges":gm["accepted_edges_directed"],"rejected_edges":gm["rejected_edges_directed"],"seconds":gt})
  for threshold in (.18,.30):
   gl,gm,gt=geometric_components(sub,subn,eps=.12,max_neighbors=24,normal_angle_degrees=35,colors=subrgb,color_distance=threshold,min_component_size=10,max_edges=500000)
   row["methods"].append({"method":"normal_plus_rgb","color_distance":threshold,"clusters":gm["components"],"assigned":int((gl>=0).sum()),"noise":gm["noise_points"],"accepted_edges":gm["accepted_edges_directed"],"rejected_edges":gm["rejected_edges_directed"],"seconds":gt})
  gl,gm,gt=geometric_components(sub,subn,eps=.12,max_neighbors=24,normal_angle_degrees=35,curvature=subcurv,curvature_tolerance=.08,min_component_size=10,max_edges=500000)
  row["methods"].append({"method":"normal_plus_pca_curvature","curvature_tolerance":.08,"clusters":gm["components"],"assigned":int((gl>=0).sum()),"noise":gm["noise_points"],"accepted_edges":gm["accepted_edges_directed"],"rejected_edges":gm["rejected_edges_directed"],"seconds":gt})
  report["spatial_cells"][name]=row
 # Plot cluster vs noise for the most populated crop using a conservative normal threshold.
 if selected:
  c=selected["dense"];ii=np.flatnonzero(inv==c);sub=p[ii];subn=nrm[ii];lab,gm,_=geometric_components(sub,subn,eps=.12,max_neighbors=24,normal_angle_degrees=35,min_component_size=10,max_edges=500000)
  fig,ax=plt.subplots(figsize=(8,7));col=np.full((len(ii),3),.25);ok=lab>=0;col[ok]=plt.cm.tab20(lab[ok]%20)[:,:3];ax.scatter(sub[:,0],sub[:,2],c=col,s=.8,linewidths=0);ax.set_title("Recorte denso: clusters geométricos; ruido gris");fig.tight_layout();fig.savefig(out/"dense_crop_after.png",dpi=180);plt.close(fig)
 report["outputs"]={"indices":str(out/"giant_indices.npz"),"before":str(out/"giant_before.png"),"context":str(out/"giant_context.png"),"after_crop":str(out/"dense_crop_after.png")};(out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
