"""Reconstruye mallas experimentales solo a partir de un ROI explícito."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np,open3d as o3d
from segmentacion.io import load_ply
from segmentacion.visualization import roi_indices
from segmentacion.reconstruction import prepare_oriented_cloud,estimate_spacing,reconstruct_bpa,reconstruct_poisson,mesh_quality,vertex_source_mapping,filter_supported_mesh

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--roi",type=float,nargs=6,required=True,metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"));p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/reconstruction/roi_run_001");p.add_argument("--mapping",type=Path,help="NPZ original_to_voxel para entrada reducida");p.add_argument("--labels",type=Path,help="NPZ segment labels alineadas al PLY de entrada");p.add_argument("--label-id",type=int,help="Filtra ROI a una sola instancia geométrica") ;p.add_argument("--seed",type=int,default=42);p.add_argument("--max-points",type=int,default=10000);p.add_argument("--allow-large",action="store_true",help="Consentimiento explícito; eleva límite absoluto a 50,000 puntos");p.add_argument("--normal-k",type=int,default=30);p.add_argument("--bpa-multipliers",type=float,nargs="+",default=[1.5,2.5,4.0]);p.add_argument("--poisson-depth",type=int,default=6);p.add_argument("--poisson-scale",type=float,default=1.1);p.add_argument("--density-quantile",type=float,default=.05);a=p.parse_args()
 try:
  cap=50000 if a.allow_large else 10000
  if a.max_points<10 or a.max_points>cap:raise ValueError(f"max-points debe estar entre 10 y {cap}; --allow-large habilita hasta 50000")
  if a.poisson_depth>7 and not a.allow_large:raise ValueError("Poisson depth > 7 requiere --allow-large")
  if a.poisson_depth>8:raise ValueError("Límite absoluto para CLI: Poisson depth <= 8")
  src=a.input.expanduser().resolve();out=a.output_dir.expanduser().resolve()
  if out.exists():raise FileExistsError(f"No se sobrescribe salida existente: {out}")
  cloud=load_ply(src);all_points=np.asarray(cloud.points);ids=roi_indices(all_points,tuple(a.roi))
  if (a.labels is None)!=(a.label_id is None):raise ValueError("--labels y --label-id deben usarse juntos")
  if a.labels:
   with np.load(a.labels) as ld:
    key="global_labels" if "global_labels" in ld.files else "original_labels" if "original_labels" in ld.files else None
    if key is None:raise ValueError("El NPZ no contiene etiquetas")
    source_labels=ld[key]
   if len(source_labels)!=len(all_points):raise ValueError("Las etiquetas no están alineadas con el PLY input")
   ids=ids[source_labels[ids]==a.label_id]
   if not len(ids):raise ValueError("ROI/label-id no seleccion? puntos")
  if len(ids)>a.max_points:raise ValueError(f"ROI tiene {len(ids)} puntos; límite={a.max_points}. Reduce ROI o autoriza --allow-large")
  crop=cloud.select_by_index(ids.tolist());prepare_start=time.perf_counter();prepared=prepare_oriented_cloud(crop,orient=True,neighbor_k=min(a.normal_k,max(3,len(ids)-1)));prepare_seconds=time.perf_counter()-prepare_start;spacing_start=time.perf_counter();spacing=estimate_spacing(prepared.cloud,seed=a.seed);spacing_seconds=time.perf_counter()-spacing_start;radii=[spacing*m for m in a.bpa_multipliers]
  out.mkdir(parents=True);reports={};meshes={}
  t=time.perf_counter();bpa,bpa_time=reconstruct_bpa(prepared.cloud,radii)
  bpa,bpa_support=filter_supported_mesh(bpa,np.asarray(prepared.cloud.points),np.asarray(prepared.cloud.normals),max_edge=4*spacing,max_distance=2*spacing)
  if bpa.has_vertex_colors():bpa.vertex_colors=o3d.utility.Vector3dVector(np.clip(np.asarray(bpa.vertex_colors),0,1))
  bpa_path=out/"ball_pivoting.ply";export_start=time.perf_counter();bpa_ok=o3d.io.write_triangle_mesh(str(bpa_path),bpa,write_vertex_colors=True);bpa_export=time.perf_counter()-export_start;reports["ball_pivoting"]={"parameters":{"radii":radii,"spacing_median":spacing},"seconds":bpa_time,"mesh":mesh_quality(bpa,np.asarray(prepared.cloud.points),seed=a.seed),"file":str(bpa_path),"file_bytes":bpa_path.stat().st_size if bpa_ok else None,"export_seconds":bpa_export,"warnings":[] if len(bpa.triangles) else ["Ball Pivoting produjo malla sin triángulos"]};meshes["bpa"]=bpa
  poisson,poisson_stats,poisson_time=reconstruct_poisson(prepared.cloud,depth=a.poisson_depth,scale=a.poisson_scale,density_quantile=a.density_quantile)
  poisson,poisson_support=filter_supported_mesh(poisson,np.asarray(prepared.cloud.points),np.asarray(prepared.cloud.normals),max_edge=4*spacing,max_distance=2*spacing)
  if poisson.has_vertex_colors():poisson.vertex_colors=o3d.utility.Vector3dVector(np.clip(np.asarray(poisson.vertex_colors),0,1))
  poisson_path=out/"poisson.ply";export_start=time.perf_counter();poisson_ok=o3d.io.write_triangle_mesh(str(poisson_path),poisson,write_vertex_colors=True);poisson_export=time.perf_counter()-export_start;reports["poisson"]={"parameters":{"depth":a.poisson_depth,"scale":a.poisson_scale,"threads":1},"seconds":poisson_time,"density_trim":poisson_stats,"mesh":mesh_quality(poisson,np.asarray(prepared.cloud.points),seed=a.seed),"file":str(poisson_path),"file_bytes":poisson_path.stat().st_size if poisson_ok else None,"export_seconds":poisson_export,"warnings":["Poisson puede cerrar huecos sin observación directa; revisar overlay"]};meshes["poisson"]=poisson
  reports["ball_pivoting"]["support_filter"]=bpa_support;reports["poisson"]["support_filter"]=poisson_support
  reports["ball_pivoting"]["geometry_provenance"]="inferred_between_observed_points";reports["poisson"]["geometry_provenance"]="inferred_poisson_surface"
  maps={};src_idx=ids[prepared.source_indices]
  correspondence=None
  if a.mapping:
   with np.load(a.mapping) as md: original_to_voxel=md["original_to_voxel"].copy();representatives=md["voxel_to_representative"].copy()
   if len(representatives)!=len(all_points) or np.any(src_idx>=len(representatives)):raise ValueError("Mapping does not align with input cloud")
   selected_mask=np.zeros(len(all_points),bool);selected_mask[ids]=True;original_ids=np.flatnonzero((original_to_voxel>=0)&selected_mask[np.maximum(original_to_voxel,0)])
   correspondence={"mapping_file":str(a.mapping.resolve()),"original_indices":original_ids,"original_to_voxel":original_to_voxel[original_ids],"representative_original_indices":representatives[src_idx]}
  for name,mesh in meshes.items():
   nearest,dist=vertex_source_mapping(mesh,np.asarray(prepared.cloud.points));maps[name+"_nearest_source_index"]=src_idx[np.maximum(nearest,0)] if len(nearest) and np.all(nearest>=0) else nearest;maps[name+"_nearest_distance"]=dist;reports["ball_pivoting" if name=="bpa" else "poisson"]["vertex_source_distance_quantiles"]=np.quantile(dist,[.5,.95,.99]).tolist() if len(dist) else []
  if correspondence is not None:maps.update({"representative_original_indices":correspondence["representative_original_indices"],"roi_original_indices":correspondence["original_indices"],"roi_original_to_voxel":correspondence["original_to_voxel"]})
  np.savez_compressed(out/"vertex_source_mapping.npz",**maps)
  report={"input":str(src),"roi_bounds":list(a.roi),"instance_filter":{"labels_file":str(a.labels.resolve()) if a.labels else None,"label_id":a.label_id},"seed":a.seed,"resource_limit":{"max_points":a.max_points,"large_run_consent":a.allow_large},"counts":{"roi_points":len(ids),"reconstruction_points":len(prepared.source_indices),"excluded_invalid_xyz":prepared.excluded_invalid_xyz,"excluded_invalid_normals":prepared.excluded_invalid_normals},"normals":{"transformation":prepared.orientation,"local_sign_consistency_before":prepared.orientation_consistency_before,"local_sign_consistency_after":prepared.orientation_consistency_after,"invalid_normals_replaced":0},"timings_seconds":{"normal_validation_orientation":prepare_seconds,"spacing_estimation":spacing_seconds,"ball_pivoting":bpa_time,"bpa_export":bpa_export,"poisson":poisson_time,"poisson_export":poisson_export},"methods":reports,"correspondence":{"file":str(out/"vertex_source_mapping.npz"),"meaning":"nearest input point per mesh vertex; exact for BPA only when vertices coincide, approximate for Poisson","source_point_indices":str(out/"roi_source_indices.npz"),"input_mapping":str(a.mapping.resolve()) if a.mapping else None},"warnings":["Only the requested spatial crop was reconstructed; not the whole factory.","Poisson may bridge gaps or close unobserved cavities.","Surface-to-point distance is one-sided and sampled; it does not establish correctness."]}
  np.savez_compressed(out/"roi_source_indices.npz",input_point_indices=ids,reconstruction_source_indices=src_idx,representative_original_indices=correspondence["representative_original_indices"] if correspondence else np.array([],dtype=np.int64),original_indices=correspondence["original_indices"] if correspondence else np.array([],dtype=np.int64));(out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
  # Static XY overlay, capped triangle draw for quick artifact inspection.
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  from matplotlib.collections import PolyCollection
  fig,axes=plt.subplots(1,2,figsize=(13,6));p0=np.asarray(prepared.cloud.points);c0=np.asarray(prepared.cloud.colors) if prepared.cloud.has_colors() else "steelblue"
  for ax,(name,mesh) in zip(axes,meshes.items()):
   ax.scatter(p0[:,0],p0[:,1],c=c0,s=1,alpha=.65);v=np.asarray(mesh.vertices);tri=np.asarray(mesh.triangles)
   if len(tri):
    if len(tri)>20000:tri=tri[np.random.default_rng(a.seed).choice(len(tri),20000,replace=False)]
    ax.add_collection(PolyCollection(v[tri][:,:,:2],facecolors="none",edgecolors="crimson",linewidths=.18,alpha=.65));ax.autoscale_view()
   ax.set_title(f"{name}: {len(mesh.triangles):,} tris");ax.set_xlabel("X");ax.set_ylabel("Y");ax.set_aspect("equal",adjustable="datalim")
  fig.tight_layout();fig.savefig(out/"overlay_xy.png",dpi=180);plt.close(fig);report["outputs"]={"directory":str(out),"overlay":str(out/"overlay_xy.png")};(out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(report,indent=2,ensure_ascii=False))
 except (OSError,ValueError,RuntimeError,MemoryError) as e:p.error(str(e))
if __name__=="__main__":main()
