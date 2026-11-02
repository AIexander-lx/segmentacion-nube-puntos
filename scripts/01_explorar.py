"""Exploración de la nube de puntos: estadísticas y figuras (Rol 1)."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentacion.io import read_ply  # noqa: E402

PLY = ROOT / "data" / "raw" / "FARO_Scan_Full_Sub1_down3_with_normals.ply"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

pc = read_ply(PLY)
x, y, z = pc.xyz.T
nz = np.abs(pc.normals[:, 2])
print(f"Puntos: {len(pc):,}")
for name, v in zip("xyz", (x, y, z)):
    print(f"{name}: min={v.min():.2f} max={v.max():.2f} media={v.mean():.2f}")
print("RGB medio:", pc.rgb.mean(0).round(1))
print("|nz|>0.9: %.1f%%  |nz|<0.2: %.1f%%" % (100 * (nz > .9).mean(), 100 * (nz < .2).mean()))

# 1. Alturas, normales y color
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(z, bins=120, color="#0072B2"); ax[0].set_xlabel("z (m)"); ax[0].set_ylabel("puntos")
ax[0].set_title("Distribución de la altura")
ax[1].hist(nz, bins=60, color="#D55E00"); ax[1].set_xlabel("|nz| de la normal")
ax[1].set_title("Verticalidad de las normales")
gray = pc.rgb.mean(1)
ax[2].hist(gray, bins=64, color="#117733"); ax[2].set_xlabel("brillo (media RGB)")
ax[2].set_title("Distribución del color")
fig.tight_layout(); fig.savefig(FIG / "01_alturas_normales_color.png", dpi=200); plt.close(fig)

# 2. Vistas: planta (x,y) y perfiles (x,z), (y,z), densidad y color real
rng = np.random.default_rng(0)
s = rng.choice(len(pc), 400_000, replace=False)
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
h = ax[0, 0].hist2d(x, y, bins=300, cmap="viridis", norm=matplotlib.colors.LogNorm())
ax[0, 0].set_title("Planta (x,y): densidad de puntos"); ax[0, 0].set_aspect("equal")
ax[0, 1].scatter(x[s], y[s], c=pc.rgb[s] / 255, s=0.2, linewidths=0)
ax[0, 1].set_title("Planta (x,y): color real"); ax[0, 1].set_aspect("equal")
ax[1, 0].scatter(x[s], z[s], c=pc.rgb[s] / 255, s=0.2, linewidths=0)
ax[1, 0].set_title("Perfil (x,z): color real"); ax[1, 0].set_aspect("equal")
ax[1, 1].scatter(x[s], z[s], c=pc.normals[s] * 0.5 + 0.5, s=0.2, linewidths=0)
ax[1, 1].set_title("Perfil (x,z): normales como color"); ax[1, 1].set_aspect("equal")
for a in ax.ravel(): a.set_xlabel("m")
fig.tight_layout(); fig.savefig(FIG / "02_vistas.png", dpi=170); plt.close(fig)
print("Figuras guardadas en", FIG)
