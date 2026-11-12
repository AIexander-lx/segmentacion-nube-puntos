"""Reproduce the existing geometry + semantics pipeline in a new directory."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from segmentacion.io import load_ply


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reuse-baseline", action="store_true", help="Exporta resultados existentes sin recalcular")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--max-global-points", type=int, default=400000)
    args = p.parse_args()
    if not 10 <= args.timeout <= 3600 or not 1 <= args.max_global_points <= 400000:
        p.error("Límites inválidos: timeout 10–3600 s y max-global-points 1–400000")
    out = args.output_dir.resolve()
    if out.exists(): p.error(f"La salida ya existe: {out}")
    original_name = "FARO_Scan_Full_Sub1_down3_with_normals.ply"
    original_candidates = (ROOT / original_name, ROOT / "data/raw" / original_name)
    original = next((path for path in original_candidates if path.is_file()), original_candidates[0])
    global_path = ROOT / "outputs/processed/global.ply"
    mapping_path = ROOT / "outputs/processed/global_mapping.npz"
    if not original.is_file():
        p.error(f"No se encontró el PLY {original_name}; descárgalo desde Canvas y colócalo en la raíz o en data/raw/")
    if not all(x.is_file() for x in (global_path, mapping_path)):
        p.error("Faltan outputs/processed/global.ply o global_mapping.npz; ejecuta primero scripts/preprocess_cloud.py")
    start = time.perf_counter()
    global_cloud = load_ply(global_path)
    if len(global_cloud.points) > args.max_global_points: p.error("La nube global supera el límite")
    del global_cloud
    out.mkdir(parents=True)
    if args.reuse_baseline:
        geometry_dir = ROOT / "outputs/segmented/global_review_final"
        semantic_dir = ROOT / "outputs/semantic/global_heuristic_v1"
    else:
        geometry_dir, semantic_dir = out / "geometry", out / "semantic"
        commands = [
            [sys.executable, str(ROOT / "scripts/segment_cloud.py"), "--output-dir", str(geometry_dir), "--seed", str(args.seed), "--eps", "0.25", "--max-neighbors", "32"],
            [sys.executable, str(ROOT / "scripts/classify_cloud.py"), "--labels", str(geometry_dir / "labels.npz"), "--planes-metrics", str(geometry_dir / "metrics.json"), "--output-dir", str(semantic_dir), "--seed", str(args.seed)],
        ]
        for command in commands:
            print("Ejecutando:", " ".join(command), flush=True)
            try: result = subprocess.run(command, cwd=ROOT, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired: p.error("Procesamiento cancelado por límite de tiempo")
            if result.returncode: p.error(f"Falló la etapa con código {result.returncode}; revisar salida parcial {out}")
    required = [geometry_dir / "labels.npz", semantic_dir / "semantic_labels.npz", semantic_dir / "semantic_original.ply"]
    if not all(path.is_file() for path in required): p.error("Faltan resultados geométricos o semánticos")
    with np.load(geometry_dir / "labels.npz") as g, np.load(semantic_dir / "semantic_labels.npz") as s, np.load(mapping_path) as m:
        mapping = m["original_to_voxel"].copy()
        instances = g["original_labels"].copy()
        classes = s["semantic_labels_original"].copy()
        if not (len(mapping) == len(instances) == len(classes)) or np.any((mapping < 0) | (mapping >= len(g["global_labels"]))):
            p.error("Etiquetas o mapeo desalineados")
        if not np.array_equal(instances, g["global_labels"][mapping]) or not np.array_equal(classes, s["semantic_labels_global"][mapping]):
            p.error("Propagación de etiquetas inconsistente")
        source = np.where(classes == 7, 2, 0).astype(np.uint8)
        np.savez_compressed(out / "labels.npz", original_to_global=mapping, instance_original=instances,
                            semantic_original=classes, semantic_global=s["semantic_labels_global"],
                            instance_global=g["global_labels"], source_original=source)
    source_cloud = load_ply(original)
    result_cloud = load_ply(semantic_dir / "semantic_original.ply")
    if len(source_cloud.points) != len(mapping) or len(result_cloud.points) != len(mapping): p.error("El PLY no coincide con los índices")
    if not np.array_equal(np.asarray(source_cloud.points), np.asarray(result_cloud.points)) or not np.array_equal(np.asarray(source_cloud.normals), np.asarray(result_cloud.normals)):
        p.error("La exportación no conserva XYZ y normales originales")
    shutil.copyfile(semantic_dir / "semantic_original.ply", out / "semantic_original.ply")
    report = {"method": "RANSAC + clustering acotado + reglas semánticas heurísticas", "reuse_baseline": args.reuse_baseline,
              "seed": args.seed, "original_points": len(mapping), "unknown_original": int(np.sum(classes == 7)),
              "semantic_coverage_percent": float(np.mean(classes != 7) * 100), "elapsed_seconds": time.perf_counter() - start,
              "outputs": {"ply": "semantic_original.ply", "npz": "labels.npz"},
              "warnings": ["Predicciones heurísticas sin anotaciones", "Z es hipótesis vertical", "No se reconstruyó la fábrica completa"],
              "provenance_codes": {"v1_or_geometric_rules": 0, "unknown": 2}, "max_global_points": args.max_global_points}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
