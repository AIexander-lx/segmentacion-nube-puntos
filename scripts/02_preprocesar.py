"""Preprocesa la nube: voxel -> limpieza -> variables -> guarda data/processed/nube_procesada.npz.

Uso:  uv run python scripts/02_preprocesar.py [--voxel 0.08] [--radio 0.25] [--min-vecinos 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentacion.io import read_ply  # noqa: E402
from segmentacion.preprocess import (FEATURE_NAMES, build_features, estimate_normals,  # noqa: E402
                                     local_geometry, outlier_mask, standardize, voxel_downsample)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", default=str(ROOT / "data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply"))
ap.add_argument("--voxel", type=float, default=0.08, help="lado del voxel (m)")
ap.add_argument("--radio", type=float, default=0.25, help="radio del filtro de ruido (m)")
ap.add_argument("--min-vecinos", type=int, default=4)
ap.add_argument("--k", type=int, default=20, help="vecinos para la geometría local")
ap.add_argument("--salida", default=str(ROOT / "data/processed/nube_procesada.npz"))
a = ap.parse_args()

pc = read_ply(a.ply)
print(f"original            : {len(pc):>9,} puntos")

vox, inverse = voxel_downsample(pc, a.voxel)
print(f"voxel {a.voxel:.2f} m       : {len(vox):>9,} puntos")

keep = outlier_mask(vox.xyz.astype(np.float64), a.radio, a.min_vecinos)
print(f"tras filtro de ruido: {keep.sum():>9,} puntos (quita {(~keep).sum():,})")
clean = vox.subset(np.flatnonzero(keep))

# La fuente trae algunas normales nulas (0,0,0): se recalculan por PCA local.
sin_normal = np.flatnonzero(np.linalg.norm(clean.normals, axis=1) < 0.5)
if len(sin_normal):
    clean.normals[sin_normal] = estimate_normals(clean.xyz.astype(np.float64), sin_normal, a.k)
    print(f"normales nulas recalculadas: {len(sin_normal)}")

geom = local_geometry(clean.xyz.astype(np.float64), a.k)
F = build_features(clean, geom)
Fs, mu, sd = standardize(F)

Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    a.salida,
    xyz=clean.xyz, rgb=clean.rgb, normals=clean.normals,
    features=F, features_std=Fs, mean=mu, std=sd,
    feature_names=np.array(FEATURE_NAMES),
    inverse=inverse, kept_to_voxel=np.flatnonzero(keep), n_voxels=len(vox),
    n_original=len(pc), voxel=a.voxel, radio=a.radio, min_vecinos=a.min_vecinos, k=a.k,
)
print("guardado en", a.salida)
