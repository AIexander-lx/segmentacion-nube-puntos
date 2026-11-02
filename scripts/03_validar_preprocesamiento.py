"""Comprueba el preprocesamiento: propagación de etiquetas y figuras de las variables."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentacion.io import read_ply  # noqa: E402
from segmentacion.preprocess import propagate_labels, voxel_downsample, outlier_mask  # noqa: E402

z = np.load(ROOT / "data/processed/nube_procesada.npz")
pc = read_ply(ROOT / "data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply")
FIG = ROOT / "outputs/figures"

# 1) Propagación: etiqueta "sobre/bajo 3 m" en la nube procesada vs. directa en la original
lab_kept = (z["xyz"][:, 2] > 3.0).astype(np.int64)
full = propagate_labels(lab_kept, z["kept_to_voxel"], z["inverse"], int(z["n_voxels"]))
direct = (pc.xyz[:, 2] > 3.0).astype(np.int64)
sin_etiqueta = full == -1
ok = full[~sin_etiqueta] == direct[~sin_etiqueta]
print(f"puntos originales sin etiqueta (ruido): {sin_etiqueta.sum():,} ({100*sin_etiqueta.mean():.3f}%)")
print(f"coincidencia propagada vs directa     : {100*ok.mean():.2f}%")

# 2) Qué se quitó como ruido (en la nube submuestreada)
vox, _ = voxel_downsample(pc, float(z["voxel"]))
keep = outlier_mask(vox.xyz.astype(np.float64), float(z["radio"]), int(z["min_vecinos"]))
rem = vox.xyz[~keep]
print("ruido removido: z min/med/max = %.2f / %.2f / %.2f" % (rem[:, 2].min(), np.median(rem[:, 2]), rem[:, 2].max()))
print("  removidos con z>9.5 m: %d de %d" % ((rem[:, 2] > 9.5).sum(), len(rem)))

# 3) Variables locales en perfil (x,z)
xyz, F, names = z["xyz"], z["features"], list(z["feature_names"])
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, nm in zip(ax.ravel()[:3], ("linealidad", "planaridad", "esfericidad")):
    sc = a.scatter(xyz[:, 0], xyz[:, 2], c=F[:, names.index(nm)], s=0.4, cmap="viridis", vmin=0, vmax=1, linewidths=0)
    a.set_title(nm + " (perfil x,z)"); a.set_aspect("equal"); plt.colorbar(sc, ax=a, shrink=0.7)
a = ax[1, 1]
a.scatter(xyz[:, 0], xyz[:, 1], c="#cccccc", s=0.2, linewidths=0)
a.scatter(rem[:, 0], rem[:, 1], c="#D55E00", s=3, linewidths=0)
a.set_title(f"Ruido removido en planta ({len(rem):,} voxeles, naranja)"); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIG / "03_variables_locales_y_ruido.png", dpi=150); plt.close(fig)
print("figura guardada")
