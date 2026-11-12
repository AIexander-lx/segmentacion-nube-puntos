"""Pipeline completo en un solo comando: PLY original -> PLY coloreado por clase.

Encadena los scripts existentes del repositorio, en este orden:
  1. preprocess_cloud.py  (voxel global/detalle + mapeos)
  2. segment_cloud.py     (planos RANSAC + componentes espaciales)
  3. classify_cloud.py    (clasificacion por reglas, V1)
  4. analyze_giant.py     (componente grande para V3)
  5. segment_v3.py        (refinamiento local por tiles, V3)
y deja el resultado final en outputs/ply/ junto con un resumen por clase.

Uso (desde la raiz del repositorio):
    uv run python scripts/run_pipeline.py --input data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply
    uv run python scripts/run_pipeline.py --input <ply> --force     # repite una corrida previa

Los parametros por defecto son los documentados en el README (seed 42).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
PROCESSED = OUT / "processed"          # analyze_giant.py lee esta ruta fija
V1_GEOM = OUT / "segmented" / "pipeline_v1"
V1_SEM = OUT / "semantic" / "pipeline_v1"
V3_SEM = OUT / "semantic" / "pipeline_v3"
GIANT = OUT / "experiments" / "giant_component_v3"   # analyze_giant.py usa esta ruta fija
FINAL_DIR = OUT / "ply"


def run(step: str, cmd: list[str], log: list[dict]) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(map(str, cmd))}", flush=True)
    start = time.perf_counter()
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    seconds = round(time.perf_counter() - start, 1)
    log.append({"step": step, "seconds": seconds, "returncode": result.returncode})
    if result.returncode != 0:
        raise SystemExit(f"Fallo el paso '{step}' (codigo {result.returncode}). Revisa el mensaje anterior.")


def class_summary(npz_path: Path) -> dict:
    """Porcentaje de puntos originales por clase a partir del NPZ de V3."""
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as data:
        labels = data["semantic_original"]
        ids = data["class_ids"].tolist()
        names = [str(n) for n in data["class_names"]]
    total = int(len(labels))
    rows = {}
    for cid, name in zip(ids, names):
        count = int((labels == cid).sum())
        rows[name] = {"points": count, "percent": round(100 * count / total, 3)}
    return {"total_points": total, "classes": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="PLY original de la fabrica")
    p.add_argument("--force", action="store_true", help="Borra salidas de una corrida anterior del pipeline")
    p.add_argument("--skip-v3", action="store_true", help="Se detiene tras V1 (mas rapido, menos cobertura)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-voxel", type=float, default=0.10)
    p.add_argument("--detail-voxel", type=float, default=0.05)
    args = p.parse_args()

    source = args.input.expanduser().resolve()
    if not source.is_file():
        p.error(f"No existe el PLY de entrada: {source}")

    stale = [d for d in (V1_GEOM, V1_SEM, V3_SEM, GIANT) if d.exists()]
    if stale and not args.force:
        p.error("Ya hay resultados de una corrida previa:\n  " + "\n  ".join(map(str, stale))
                + "\nUsa --force para borrarlos y repetir (los scripts no sobrescriben).")
    for d in stale:
        shutil.rmtree(d)

    s = str(args.seed)
    log: list[dict] = []
    scripts = ROOT / "scripts"
    t0 = time.perf_counter()

    run("1/5 Preprocesamiento (voxel)", [PY, scripts / "preprocess_cloud.py", "--input", source,
        "--global-voxel", args.global_voxel, "--detail-voxel", args.detail_voxel,
        "--output-dir", PROCESSED, "--seed", s], log)
    g_cloud, g_map = PROCESSED / "global.ply", PROCESSED / "global_mapping.npz"

    run("2/5 Planos RANSAC + clustering espacial", [PY, scripts / "segment_cloud.py", "--original", source,
        "--global-cloud", g_cloud, "--mapping", g_map, "--output-dir", V1_GEOM,
        "--seed", s, "--eps", "0.25", "--max-neighbors", "64"], log)

    run("3/5 Clasificacion por reglas (V1)", [PY, scripts / "classify_cloud.py", "--original", source,
        "--global-cloud", g_cloud, "--mapping", g_map, "--labels", V1_GEOM / "labels.npz",
        "--planes-metrics", V1_GEOM / "metrics.json", "--output-dir", V1_SEM, "--seed", s], log)

    final_ply, npz = V1_SEM / "semantic_original.ply", None
    if not args.skip_v3:
        run("4/5 Componente grande para V3", [PY, scripts / "analyze_giant.py"], log)
        run("5/5 Refinamiento local por tiles (V3)", [PY, scripts / "segment_v3.py", "--original", source,
            "--global-cloud", g_cloud, "--mapping", g_map,
            "--semantic-v1", V1_SEM / "semantic_labels.npz", "--geometry-v1", V1_GEOM / "labels.npz",
            "--giant-indices", GIANT / "giant_indices.npz", "--output-dir", V3_SEM,
            "--tile-size", "2.0", "--halo", "0.20", "--voxel-size", "0.20", "--normal-angle", "35",
            "--min-region-points", "20", "--max-tile-points", "5000", "--max-edges", "100000",
            "--max-tiles", "5000", "--max-context-points", "2000000", "--max-global-points", "400000",
            "--max-original-points", "2000000", "--max-memory-mb", "1024", "--seed", s], log)
        final_ply, npz = V3_SEM / "semantic_v3_original.ply", V3_SEM / "semantic_v3_labels.npz"

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    target = FINAL_DIR / "fabrica_segmentada.ply"
    shutil.copyfile(final_ply, target)

    summary = {"input": str(source), "final_ply": str(target), "version": "V1" if args.skip_v3 else "V3",
               "seed": args.seed, "total_seconds": round(time.perf_counter() - t0, 1), "steps": log}
    if npz is not None:
        summary["class_summary"] = class_summary(npz)
    summary_path = OUT / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nListo. PLY coloreado: {target}\nResumen por clase: {summary_path}")
    if "class_summary" in summary:
        for name, row in summary["class_summary"]["classes"].items():
            print(f"  {name:8s} {row['percent']:7.3f} %  ({row['points']:,} puntos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
