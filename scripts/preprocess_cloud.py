"""Genera resoluciones global y de detalle sin modificar el PLY de entrada."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import open3d as o3d
from segmentacion.io import load_ply
from segmentacion.preprocessing import voxel_reduce

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--global-voxel",type=float,default=0.10)
    parser.add_argument("--detail-voxel",type=float,default=0.05)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/processed")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--visualize",action="store_true",help="Abre Open3D GUI al final (opcional)")
    args=parser.parse_args()
    try:
        if args.global_voxel<=0 or args.detail_voxel<=0 or args.global_voxel<=args.detail_voxel: raise ValueError("Use vóxeles positivos y global-voxel > detail-voxel")
        source=args.input.expanduser().resolve(); outdir=args.output_dir.expanduser().resolve()
        if source in (outdir/"global.ply", outdir/"detail.ply"): raise ValueError("La salida coincidiría con el archivo de entrada")
        cloud=load_ply(source)
        runs={}
        for name,size in (("global",args.global_voxel),("detail",args.detail_voxel)):
            result=voxel_reduce(cloud,size); ply=outdir/f"{name}.ply"; mapping=outdir/f"{name}_mapping.npz"
            outdir.mkdir(parents=True,exist_ok=True)
            if not o3d.io.write_point_cloud(str(ply),result.cloud): raise OSError(f"No se pudo escribir {ply}")
            np.savez_compressed(mapping,original_to_voxel=result.original_to_voxel,voxel_to_representative=result.voxel_to_representative,normal_valid=result.normal_valid)
            runs[name]={**result.stats,"ply":str(ply),"mapping":str(mapping)}
        report={"input":str(source),"seed":args.seed,"versions":{"python":sys.version.split()[0],"numpy":np.__version__,"open3d":o3d.__version__},"strategy":"Resoluciones independientes; no concatenar global y detalle","mapping":"original_to_voxel asigna cada fila fuente a voxel; -1 para XYZ inválido. voxel_to_representative identifica punto fuente más cercano al centroide.","runs":runs}
        metric=outdir/"preprocessing_metrics.json"; metric.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        figpath=ROOT/"outputs/figures/preprocessing_comparison.png"; figpath.parent.mkdir(parents=True,exist_ok=True)
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        points=np.asarray(cloud.points); ids=np.random.default_rng(args.seed).choice(len(points),min(100000,len(points)),replace=False)
        colors=np.asarray(cloud.colors)[ids] if cloud.has_colors() else "steelblue"
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        for row,(title,pc) in enumerate((("Global",runs["global"]),("Detalle",runs["detail"]))):
            reduced=o3d.io.read_point_cloud(pc["ply"]); all_p=np.asarray(reduced.points)
            sample_ids=np.random.default_rng(args.seed).choice(len(all_p),min(100000,len(all_p)),replace=False); p=all_p[sample_ids]; c=np.asarray(reduced.colors)[sample_ids] if reduced.has_colors() else "steelblue"
            for ax,(i,j,plane) in zip(axes[row],((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))):
                if row==0: ax.scatter(points[ids,i],points[ids,j],c=colors,s=.12,linewidths=0,rasterized=True)
                ax.scatter(p[:,i],p[:,j],c=c,s=.18,linewidths=0,rasterized=True)
                ax.set_title(f"{title} ? {plane} ({len(p):,} pts)"); ax.set_xlabel("XYZ"[i]); ax.set_ylabel("XYZ"[j]); ax.set_aspect("equal",adjustable="datalim")
        fig.tight_layout(); fig.savefig(figpath,dpi=180); plt.close(fig)
        print(json.dumps(report,indent=2,ensure_ascii=False)); print(f"Figura comparativa: {figpath}")
        if args.visualize: o3d.visualization.draw_geometries([o3d.io.read_point_cloud(runs["global"]["ply"]),o3d.io.read_point_cloud(runs["detail"]["ply"])])
        return 0
    except (OSError,ValueError) as exc: parser.error(str(exc)); return 2
if __name__=="__main__": raise SystemExit(main())
