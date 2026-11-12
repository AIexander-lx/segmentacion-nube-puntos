"""Comparacion de algoritmos de clustering sobre las variables estandarizadas.

Uso: uv run python scripts/04_comparar_clustering.py [--npz RUTA] [--n 10000]

Muestrea n puntos de la nube procesada (3 semillas), ejecuta k-means, mezclas
gaussianas, jerarquico (Ward) y DBSCAN sobre las 13 variables estandarizadas
y reporta silhouette, Davies-Bouldin y tiempo. Guarda tablas y figuras en
outputs/clustering/.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

p = argparse.ArgumentParser()
p.add_argument("--npz", default="data/processed/nube_procesada.npz")
p.add_argument("--n", type=int, default=10000)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--out", default="outputs/clustering")
p.add_argument("--sin-posicion", action="store_true", help="excluye x,y,z de las variables (10 variables)")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

d = np.load(a.npz, allow_pickle=True)
Xs = d["features_std"].astype(np.float64)
names = [str(s) for s in d["feature_names"]]
xyz = d["xyz"]
if a.sin_posicion:
    Xs = Xs[:, 3:]; names = names[3:]
ks = list(range(2, 11))

def metrics(X, lab):
    m = lab >= 0
    if len(set(lab[m])) < 2 or m.sum() < 10:
        return np.nan, np.nan
    return silhouette_score(X[m], lab[m]), davies_bouldin_score(X[m], lab[m])

def run(method, X, k, seed):
    t = time.time()
    if method == "kmeans":
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    elif method == "gmm":
        lab = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed).fit(X).predict(X)
    else:
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    return lab, time.time() - t

res = {}  # (method,k) -> list of (sil, db, t)
for s in a.seeds:
    rng = np.random.default_rng(s)
    idx = rng.choice(len(Xs), a.n, replace=False)
    X = Xs[idx]
    for m in ("kmeans", "gmm", "ward"):
        for k in ks:
            lab, t = run(m, X, k, s)
            sil, db = metrics(X, lab)
            res.setdefault((m, k), []).append((sil, db, t))
    # DBSCAN: barrido de eps con min_samples=10
    for eps in (0.5, 0.75, 1.0, 1.5, 2.0):
        t = time.time(); lab = DBSCAN(eps=eps, min_samples=10).fit_predict(X); t = time.time() - t
        sil, db = metrics(X, lab)
        res.setdefault(("dbscan", eps), []).append((sil, db, t, len(set(lab) - {-1}), float((lab == -1).mean())))
    print("semilla", s, "lista", flush=True)

rows = []
for (m, k), v in res.items():
    v = np.array(v, dtype=float)
    r = dict(metodo=m, param=k, silhouette=float(np.nanmean(v[:, 0])), sil_std=float(np.nanstd(v[:, 0])),
             davies_bouldin=float(np.nanmean(v[:, 1])), tiempo_s=float(v[:, 2].mean()))
    if m == "dbscan":
        r["n_clusters"] = float(v[:, 3].mean()); r["ruido"] = float(v[:, 4].mean())
    rows.append(r)
json.dump(rows, open(out / "comparacion.json", "w"), indent=1)

# tabla resumen: mejor k por silhouette para cada metodo
print("\nmetodo  mejor_param  silhouette(+-std)  DB  tiempo")
for r in rows:
    if r["metodo"] == "dbscan":  # DBSCAN: se reporta todo el barrido (la silhouette ignora el ruido)
        print("dbscan eps", r["param"], f'sil={r["silhouette"]:.3f} DB={r["davies_bouldin"]:.2f} clusters={r.get("n_clusters")} ruido={r.get("ruido"):.2f}')
for m in ("kmeans", "gmm", "ward"):
    rs = [r for r in rows if r["metodo"] == m and not np.isnan(r["silhouette"])]
    b = max(rs, key=lambda r: r["silhouette"])
    print(m, b["param"], f'{b["silhouette"]:.3f}+-{b["sil_std"]:.3f}', f'{b["davies_bouldin"]:.2f}', f'{b["tiempo_s"]:.1f}s',
          b.get("n_clusters", ""), b.get("ruido", ""))

# figura 1: silhouette vs k
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for m, lbl in (("kmeans", "k-means"), ("gmm", "GMM (diag.)"), ("ward", "Jerárquico (Ward)")):
    ys = [next(r for r in rows if r["metodo"] == m and r["param"] == k) for k in ks]
    ax[0].errorbar(ks, [y["silhouette"] for y in ys], yerr=[y["sil_std"] for y in ys], marker="o", label=lbl, capsize=2)
    ax[1].plot(ks, [y["davies_bouldin"] for y in ys], marker="o", label=lbl)
ax[0].set_xlabel("k"); ax[0].set_ylabel("Silhouette (mayor es mejor)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("k"); ax[1].set_ylabel("Davies-Bouldin (menor es mejor)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(out / "silhouette_vs_k.png", dpi=200); plt.close()

# figura 2: planta XY coloreada por k-means y por GMM con el mejor k, semilla 0
rng = np.random.default_rng(0); idx = rng.choice(len(Xs), a.n, replace=False)
X = Xs[idx]
best = {}
for m in ("kmeans", "gmm", "ward"):
    rs = [r for r in rows if r["metodo"] == m]
    best[m] = int(max(rs, key=lambda r: r["silhouette"])["param"])
big = rng.choice(len(Xs), 60000, replace=False)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for j, (m, lbl) in enumerate((("kmeans", "k-means"), ("gmm", "GMM"), ("ward", "Ward"))):
    k = best[m]
    if m == "kmeans":
        model = KMeans(k, n_init=10, random_state=0).fit(X); lab = model.predict(Xs[big])
    elif m == "gmm":
        model = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=0).fit(X); lab = model.predict(Xs[big])
    else:  # Ward no predice: asigna al centroide mas cercano de la muestra
        l0 = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
        C = np.array([X[l0 == c].mean(0) for c in range(k)])
        lab = ((Xs[big][:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    ax[j].scatter(xyz[big, 0], xyz[big, 1], c=lab, s=0.6, cmap="tab10")
    ax[j].set_title(f"{lbl}, k={k}"); ax[j].set_aspect("equal"); ax[j].set_xlabel("x (m)"); ax[j].set_ylabel("y (m)")
plt.tight_layout(); plt.savefig(out / "planta_clusters.png", dpi=200); plt.close()

# figura 3: perfil medio de variables por cluster de k-means (mejor k) para interpretar
k = best["kmeans"]; km = KMeans(k, n_init=10, random_state=0).fit(X)
lab = km.labels_
perf = np.array([d["features"][idx][:, -len(names):][lab == c].mean(0) for c in range(k)])
sizes = np.bincount(lab, minlength=k)
np.savetxt(out / "kmeans_perfiles.csv", np.c_[sizes, perf], delimiter=",", header="n," + ",".join(names), comments="")
print("\nperfil k-means k=", k, "tamanos", sizes.tolist())
print("best", best)
