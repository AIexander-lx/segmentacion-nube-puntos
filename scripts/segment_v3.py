"""Run a bounded V3 semantic refinement on the saved giant component."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from segmentacion.io import load_ply
from segmentacion.semantic_classifier import CLASS_IDS, RuleConfig
from segmentacion.semantic_v3 import estimate_peak_memory_bytes, export_v3, peak_working_set_bytes, refine_component


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "FARO_Scan_Full_Sub1_down3_with_normals.ply")
    parser.add_argument("--global-cloud", type=Path, default=ROOT / "outputs/processed/global.ply")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/processed/global_mapping.npz")
    parser.add_argument("--semantic-v1", type=Path, default=ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz")
    parser.add_argument("--geometry-v1", type=Path, default=ROOT / "outputs/segmented/global_review_final/labels.npz")
    parser.add_argument("--giant-indices", type=Path, default=ROOT / "outputs/experiments/giant_component_v3/giant_indices.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/semantic/global_local_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=float, default=2.0)
    parser.add_argument("--halo", type=float, default=0.20)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--normal-angle", type=float, default=35.0)
    parser.add_argument("--min-region-points", type=int, default=20)
    parser.add_argument("--max-tile-points", type=int, default=5_000)
    parser.add_argument("--max-edges", type=int, default=100_000)
    parser.add_argument("--max-tiles", type=int, default=5_000)
    parser.add_argument("--max-context-points", type=int, default=2_000_000)
    parser.add_argument("--max-memory-mb", type=int, default=1_024, help="Preflight conservador; no impone un límite de RAM del SO")
    parser.add_argument("--max-global-points", type=int, default=400_000)
    parser.add_argument("--max-original-points", type=int, default=2_000_000)
    parser.add_argument("--decision-min-score", type=float, default=0.62)
    parser.add_argument("--decision-margin", type=float, default=0.12)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"No se sobrescriben resultados: {args.output_dir}")
    if not (1 <= args.max_global_points <= 400_000 and 1 <= args.max_original_points <= 2_000_000 and args.max_memory_mb >= 256):
        parser.error("Límites global/original inválidos")
    if min(args.max_tile_points, args.max_edges, args.max_tiles, args.max_context_points, args.min_region_points) < 1:
        parser.error("Los límites de tile, aristas, contexto y región deben ser positivos")
    if not (args.tile_size > 0 and args.voxel_size > 0 and 0 <= args.halo < args.tile_size):
        parser.error("Se requiere tile_size/voxel_size positivos y 0 <= halo < tile_size")
    if not 0 <= args.normal_angle <= 90:
        parser.error("normal-angle debe estar entre 0 y 90 grados")
    if not (0 <= args.decision_min_score <= 1 and 0 <= args.decision_margin <= 1):
        parser.error("Los umbrales heurísticos deben estar entre 0 y 1")
    for path in (args.original, args.global_cloud, args.mapping, args.semantic_v1, args.geometry_v1, args.giant_indices):
        if not path.is_file():
            parser.error(f"No existe el archivo requerido: {path}")

    started = time.perf_counter()
    original = load_ply(args.original)
    global_cloud = load_ply(args.global_cloud)
    xyz = np.asarray(global_cloud.points)
    if len(xyz) > args.max_global_points:
        parser.error(f"Global excede el límite de {args.max_global_points} puntos")
    if len(original.points) > args.max_original_points:
        parser.error(f"Original excede el límite de {args.max_original_points} puntos")
    estimated_memory = estimate_peak_memory_bytes(
        len(original.points), len(xyz), max_tile_points=args.max_tile_points,
        max_edges=args.max_edges, max_context_points=args.max_context_points,
    )
    memory_budget = args.max_memory_mb * 1024 * 1024
    if estimated_memory > memory_budget:
        parser.error(
            f"Preflight estima {estimated_memory / 1024**2:.0f} MB; supera --max-memory-mb={args.max_memory_mb}"
        )
    if not global_cloud.has_normals() or not global_cloud.has_colors():
        parser.error("La nube global requiere RGB y normales")
    with np.load(args.mapping) as data:
        original_to_global = data["original_to_voxel"].copy()
        representative = data["voxel_to_representative"].copy()
    with np.load(args.semantic_v1) as data:
        v1_global = data["semantic_labels_global"].copy()
        v1_original = data["semantic_labels_original"].copy()
    with np.load(args.geometry_v1) as data:
        geometry_global = data["global_labels"].copy()
    with np.load(args.giant_indices) as data:
        giant_indices = data["global_indices"].copy()
    if len(representative) != len(xyz) or np.any((representative < 0) | (representative >= len(original_to_global))):
        parser.error("voxel_to_representative no coincide con la nube global")
    if not np.array_equal(original_to_global[representative], np.arange(len(xyz), dtype=np.int64)):
        parser.error("Los representantes no apuntan a su voxel global")
    if np.any(geometry_global[giant_indices] != -1):
        parser.error("El archivo de componente contiene puntos que no son ruido geométrico baseline")
    load_seconds = time.perf_counter() - started

    config = RuleConfig(
        decision_min_score=args.decision_min_score,
        decision_margin=args.decision_margin,
        require_tube_radial_evidence=True,
    )
    arrays, report = refine_component(
        xyz, np.asarray(global_cloud.normals), np.asarray(global_cloud.colors),
        v1_global, v1_original, original_to_global, giant_indices,
        instance_v1_global=geometry_global,
        tile_size=args.tile_size, halo=args.halo, voxel_size=args.voxel_size,
        normal_angle_degrees=args.normal_angle, min_region_points=args.min_region_points,
        max_tile_points=args.max_tile_points, max_edges=args.max_edges,
        max_tiles=args.max_tiles, max_context_points=args.max_context_points,
        config=config,
    )
    arrays["semantic_v1_global"] = v1_global.astype(np.int8, copy=True)
    arrays["semantic_v1_original"] = v1_original.astype(np.int8, copy=True)
    arrays["instance_v1_global"] = geometry_global.astype(np.int64, copy=True)
    report["timings_seconds"]["load_and_alignment"] = load_seconds
    report["memory_preflight"] = {
        "estimated_peak_bytes": estimated_memory,
        "budget_bytes": memory_budget,
        "within_configured_budget": estimated_memory <= memory_budget,
        "note": "Estimación conservadora de buffers y reserva; el sistema operativo no impone este límite.",
    }
    report["comparison_v1"] = {
        "global_unknown": int(np.sum(v1_global == CLASS_IDS["unknown"])),
        "original_unknown": int(np.sum(v1_original == CLASS_IDS["unknown"])),
        "global_coverage_percent": float(np.mean(v1_global != CLASS_IDS["unknown"]) * 100),
        "original_coverage_percent": float(np.mean(v1_original != CLASS_IDS["unknown"]) * 100),
    }
    report["comparison_v3"] = {
        "global_coverage_percent": float(np.mean(arrays["semantic_global"] != CLASS_IDS["unknown"]) * 100),
        "original_coverage_percent": float(np.mean(arrays["semantic_original"] != CLASS_IDS["unknown"]) * 100),
        "newly_proposed_original": int(np.sum(arrays["source_original"] == 1)),
        "changed_v1_known_original": int(np.sum(
            (v1_original != CLASS_IDS["unknown"])
            & (arrays["semantic_original"] != v1_original)
        )),
        "new_class_counts_original": {
            name: int(np.sum((arrays["source_original"] == 1) & (arrays["semantic_original"] == class_id)))
            for name, class_id in CLASS_IDS.items()
        },
    }
    export_started = time.perf_counter()
    out_paths = export_v3(args.original, args.output_dir, arrays, report, seed=args.seed)
    # The export routine writes a color-only copy. Re-read and prove that the
    # measured geometry and normals are identical to the original file.
    exported = load_ply(out_paths["ply"])
    if len(exported.points) != len(original.points):
        raise RuntimeError("El PLY semántico no conserva el número original de puntos")
    if not np.array_equal(np.asarray(exported.points), np.asarray(original.points)):
        raise RuntimeError("El PLY semántico modificó coordenadas originales")
    if original.has_normals() != exported.has_normals() or not np.array_equal(np.asarray(exported.normals), np.asarray(original.normals)):
        raise RuntimeError("El PLY semántico modificó normales originales")
    report["integrity"] = {
        "original_points_preserved": len(exported.points),
        "xyz_equal_to_source": True,
        "normals_equal_to_source": True,
        "semantic_labels_original_aligned": len(arrays["semantic_original"]) == len(exported.points),
        "mapping_original_to_global_exact": True,
    }
    report["timings_seconds"]["export_and_figure"] = time.perf_counter() - export_started
    report["timings_seconds"]["end_to_end_including_export_validation"] = time.perf_counter() - started
    measured_peak = peak_working_set_bytes()
    if measured_peak is not None:
        report["peak_working_set_bytes"] = max(report.get("peak_working_set_bytes") or 0, measured_peak)
    metrics = Path(out_paths["metrics"])
    metrics.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"comparison_v1": report["comparison_v1"], "comparison_v3": report["comparison_v3"], "counts": report["counts"], "outputs": out_paths, "integrity": report["integrity"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
