"""Visualiza original/global/detail y segmentaciones sin abrir GUI por defecto."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np,open3d as o3d
from segmentacion.io import load_ply
from segmentacion.visualization import label_colors,normals_colors,roi_indices,save_roi,labels_for_resolution
from segmentacion.evaluation import internal_summary,compare_runs

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--resolution",choices=("original","global","detail"),default="original");p.add_argument("--color",choices=("rgb","normals","segments"),default="rgb");p.add_argument("--labels",type=Path,help="Archivo labels.npz (requerido para --color segments)");p.add_argument("--preset",choices=("baseline","eps025"),help="Corridas guardadas: baseline 3A o eps=.25 k=64");p.add_argument("--compare-labels",type=Path,nargs="*",default=[]);p.add_argument("--roi",type=float,nargs=6,metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"));p.add_argument("--save-roi",type=Path);p.add_argument("--figure",type=Path);p.add_argument("--mesh",type=Path,help="Malla PLY opcional, superpuesta a la nube");p.add_argument("--camera",choices=("top","front","side","iso"),default="iso");p.add_argument("--point-size",type=float,default=2.0);p.add_argument("--lighting",action="store_true");p.add_argument("--show",action="store_true",help="Abre ventana interactiva Open3D");a=p.parse_args()
 try:
  cloud_path={"original":ROOT/"FARO_Scan_Full_Sub1_down3_with_normals.ply","global":ROOT/"outputs/processed/global.ply","detail":ROOT/"outputs/processed/detail.ply"}[a.resolution];map_path={"global":ROOT/"outputs/processed/global_mapping.npz","detail":ROOT/"outputs/processed/detail_mapping.npz"}.get(a.resolution)
  cloud=load_ply(cloud_path);points=np.asarray(cloud.points);representatives=None;orig_to_voxel=None
  if map_path:
   with np.load(map_path) as mp:representatives=mp["voxel_to_representative"].copy();orig_to_voxel=mp["original_to_voxel"].copy()
  original_count=len(points) if orig_to_voxel is None else len(orig_to_voxel)
  labels_path=a.labels
  if a.preset:labels_path=ROOT/"outputs/segmented"/("global_seed42" if a.preset=="baseline" else "global_review_final")/"labels.npz"
  if a.color=="segments" and labels_path is None:raise ValueError("--color segments requiere --labels o --preset")
  labels=labels_for_resolution(labels_path,a.resolution,original_count,len(points),map_path) if labels_path else None
  colors=np.asarray(cloud.colors).copy() if a.color=="rgb" and cloud.has_colors() else normals_colors(np.asarray(cloud.normals)) if a.color=="normals" and cloud.has_normals() else label_colors(labels) if a.color=="segments" else np.full((len(points),3),.5)
  if a.color=="rgb" and not cloud.has_colors():raise ValueError("La nube no contiene RGB")
  if a.color=="normals" and not cloud.has_normals():raise ValueError("La nube no contiene normales")
  selected=roi_indices(points,tuple(a.roi)) if a.roi else np.arange(len(points));view=cloud.select_by_index(selected.tolist());view.colors=o3d.utility.Vector3dVector(colors[selected])
  mesh_obj=None
  if a.mesh:
   mesh_obj=o3d.io.read_triangle_mesh(str(a.mesh))
   if mesh_obj.is_empty():raise ValueError(f"Malla vacía o ilegible: {a.mesh}")
  output={"resolution":a.resolution,"point_count":len(points),"visible_points":len(selected),"color_mode":a.color,"source":str(cloud_path)}
  if labels is not None:output["metrics"]=internal_summary(labels[selected])
  if a.save_roi:
   if not a.roi:raise ValueError("--save-roi requiere --roi")
   roi_path=a.save_roi if a.save_roi.is_absolute() else ROOT/a.save_roi
   save_roi(points,selected,roi_path,resolution=a.resolution,representative_original=representatives,original_to_voxel=orig_to_voxel if a.resolution!="original" else None);output["roi_file"]=str(roi_path.resolve())
  if a.compare_labels:
   if labels is None:raise ValueError("Comparación requiere --color segments y una corrida primaria")
   output["comparisons"]=[]
   for path in a.compare_labels:
    other=labels_for_resolution(path,a.resolution,original_count,len(points),map_path);output["comparisons"].append({"path":str(path.resolve()),**compare_runs(labels[selected],other[selected])})
  if a.figure:
   import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
   fig,axs=plt.subplots(1,3,figsize=(15,5))
   if mesh_obj is not None:from matplotlib.collections import PolyCollection;mesh_vertices=np.asarray(mesh_obj.vertices);mesh_triangles=np.asarray(mesh_obj.triangles);mesh_triangles=mesh_triangles[np.random.default_rng(42).choice(len(mesh_triangles),min(20000,len(mesh_triangles)),replace=False)] if len(mesh_triangles)>20000 else mesh_triangles
   for ax,(i,j,title) in zip(axs,((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))):
    ax.scatter(points[selected,i],points[selected,j],c=colors[selected],s=.15,linewidths=0,rasterized=True)
    if mesh_obj is not None and len(mesh_triangles):
     ax.add_collection(PolyCollection(mesh_vertices[mesh_triangles][:,:,[i,j]],facecolors="none",edgecolors="crimson",linewidths=.2,alpha=.7));ax.autoscale_view()
    ax.set_title(title);ax.set_xlabel("XYZ"[i]);ax.set_ylabel("XYZ"[j]);ax.set_aspect("equal",adjustable="datalim")
   fig.tight_layout();dest=a.figure if a.figure.is_absolute() else ROOT/a.figure;dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=180);plt.close(fig);output["figure"]=str(dest.resolve())
  if a.mesh:output["mesh"]=str(a.mesh.resolve())
  print(json.dumps(output,indent=2,ensure_ascii=False))
  if a.show:
   if a.point_size<=0:raise ValueError("--point-size debe ser positivo")
   vis=o3d.visualization.Visualizer();vis.create_window(window_name=f"Nube {a.resolution} - {a.color}")
   vis.add_geometry(view);mesh=mesh_obj
   if mesh is not None:
    if not mesh.has_vertex_colors():mesh.paint_uniform_color([.85,.18,.12])
    vis.add_geometry(mesh)
   options=vis.get_render_option();options.point_size=a.point_size;options.light_on=a.lighting;options.mesh_show_back_face=True
   control=vis.get_view_control();bounds=np.asarray(view.points);center=bounds.mean(axis=0);control.set_lookat(center.tolist())
   camera={"top":([0,0,-1],[0,1,0]),"front":([0,-1,0],[0,0,1]),"side":([-1,0,0],[0,0,1]),"iso":([1,-1,.7],[0,0,1])}[a.camera]
   control.set_front(camera[0]);control.set_up(camera[1]);control.set_zoom(.65);vis.run();vis.destroy_window()
 except (OSError,ValueError,KeyError) as e:p.error(str(e))
if __name__=="__main__":main()
