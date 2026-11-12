"""Run a bounded V1 plus local-voxel experiment on one contiguous XYZ ROI."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from segmentacion.hybrid_pipeline import (
    SOURCE_IDS,
    fuse_unknown_only,
    make_semantic_cloud,
    project_roi_to_original,
    select_roi_indices,
)
from segmentacion.io import load_ply
from segmentacion.local_regions import local_voxel_regions
from segmentacion.semantic_classifier import (
    CLASS_IDS,
    RuleConfig,
    classify_region,
    rule_config_dict,
    semantic_colors,
)
from segmentacion.semantic_features import region_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/processed/global.ply")
    parser.add_argument("--geometry-labels", type=Path, default=ROOT / "outputs/segmented/global_review_final/labels.npz")
    parser.add_argument("--semantic-v1", type=Path, default=ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/processed/global_mapping.npz")
    parser.add_argument("--planes-metrics", type=Path, default=ROOT / "outputs/segmented/global_review_final/metrics.json")
    parser.add_argument("--roi", type=float, nargs=6, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/hybrid_roi_run_001")
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--normal-angle", type=float, default=35.0)
    parser.add_argument("--min-region-points", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=5_000)
    parser.add_argument("--max-edges", type=int, default=100_000)
    parser.add_argument("--decision-min-score", type=float, default=0.70)
    parser.add_argument("--vertical-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--vertical-confirmed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(labels == class_id)) for name, class_id in CLASS_IDS.items()}


def percent(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return 100.0 * float(np.mean(values)) if len(values) else 0.0


def region_colors(labels: np.ndarray) -> np.ndarray:
    """Stable visualization colors for the local region IDs and noise."""
    ids = np.asarray(labels)
    colors = np.full((len(ids), 3), 0.45)
    valid = ids >= 0
    if np.any(valid):
        color_map = plt.get_cmap("turbo")
        colors[valid] = color_map((ids[valid] * 0.61803398875) % 1.0)[:, :3]
    return colors


def make_comparison_figure(
    path: Path,
    xyz: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    regions: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, colors, title in (
        (axes[0], semantic_colors(v1), "V1 original"),
        (axes[1], semantic_colors(v2), "V2 híbrida"),
        (axes[2], region_colors(regions), "Regiones voxel (no clases)"),
    ):
        axis.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=5, linewidths=0, rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle("Mismos puntos globales del ROI; gris = unknown/ruido")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescriben resultados: {output_dir}")
    bounds = (np.asarray(args.roi[::2]), np.asarray(args.roi[1::2]))
    if args.max_points > 5_000:
        raise ValueError("El CLI experimental limita max-points a 5,000 por ejecución")

    timers: dict[str, float] = {}
    started = time.perf_counter()
    global_cloud = load_ply(args.input)
    points = np.asarray(global_cloud.points)
    normals = np.asarray(global_cloud.normals) if global_cloud.has_normals() else np.zeros_like(points)
    colors = np.asarray(global_cloud.colors) if global_cloud.has_colors() else np.zeros_like(points)
    with np.load(args.geometry_labels) as stored:
        geometry_labels = stored["global_labels"].copy()
    with np.load(args.semantic_v1) as stored:
        semantic_v1_all = stored["semantic_labels_global"].copy()
        semantic_v1_original = stored["semantic_labels_original"].copy()
    with np.load(args.mapping) as stored:
        original_to_global = stored["original_to_voxel"].copy()
    planes = json.loads(args.planes_metrics.read_text(encoding="utf-8")).get("planes", [])
    timers["input_loading_seconds"] = time.perf_counter() - started

    if len(geometry_labels) != len(points) or len(semantic_v1_all) != len(points):
        raise ValueError("Las etiquetas globales no coinciden con la nube global")
    if len(original_to_global) != len(semantic_v1_original):
        raise ValueError("El mapeo y las etiquetas semánticas originales tienen longitudes distintas")
    if not np.issubdtype(original_to_global.dtype, np.integer):
        raise ValueError("original_to_voxel debe contener índices enteros")
    in_range = (original_to_global >= 0) & (original_to_global < len(points))
    roi_indices = select_roi_indices(points, bounds, max_points=args.max_points)
    roi_points = points[roi_indices]
    roi_normals = normals[roi_indices]
    roi_rgb = colors[roi_indices]
    roi_v1 = semantic_v1_all[roi_indices]
    roi_geometry = geometry_labels[roi_indices]

    original_counts = np.bincount(original_to_global[in_range], minlength=len(points))
    mapping_valid_global = original_counts[roi_indices] > 0
    mismatch = np.zeros(len(original_to_global), dtype=bool)
    mismatch[in_range] = (
        semantic_v1_original[in_range] != semantic_v1_all[original_to_global[in_range]]
    )
    mismatch_per_global = np.bincount(original_to_global[mismatch], minlength=len(points))
    mapping_valid_global &= mismatch_per_global[roi_indices] == 0

    started = time.perf_counter()
    local_regions, region_metrics, region_seconds = local_voxel_regions(
        roi_points,
        roi_normals,
        voxel_size=args.voxel_size,
        normal_angle_degrees=args.normal_angle,
        min_region_points=args.min_region_points,
        max_points=args.max_points,
        max_edges=args.max_edges,
    )
    timers["local_region_extraction_seconds"] = region_seconds

    rule_config = RuleConfig(decision_min_score=args.decision_min_score)
    local_proposals = np.full(len(roi_indices), CLASS_IDS["unknown"], dtype=np.int8)
    region_records = []
    scene_bounds = (points.min(axis=0), points.max(axis=0))
    feature_started = time.perf_counter()
    for region_id in np.unique(local_regions[local_regions >= 0]):
        rows = np.flatnonzero(local_regions == region_id)
        features = region_features(
            roi_points[rows],
            normals=roi_normals[rows],
            colors=roi_rgb[rows],
            scene_bounds=scene_bounds,
            vertical_axis="xyz".index(args.vertical_axis),
            plane_models=planes,
        )
        prediction = classify_region(
            features,
            region_type="cluster",
            vertical_axis_confirmed=args.vertical_confirmed,
            config=rule_config,
        )
        local_proposals[rows] = prediction["class_id"]
        region_records.append({
            "region_id_local": int(region_id),
            "point_count": int(len(rows)),
            "features_valid": bool(features.get("valid", False)),
            "features": features,
            "prediction": prediction,
        })
    timers["feature_and_classification_seconds"] = time.perf_counter() - feature_started

    started = time.perf_counter()
    hybrid_global, source_global, fusion_metrics = fuse_unknown_only(
        roi_v1,
        local_proposals,
        local_regions,
        mapping_valid=mapping_valid_global,
    )
    projected = project_roi_to_original(
        roi_global_indices=roi_indices,
        hybrid_roi_labels=hybrid_global,
        hybrid_roi_sources=source_global,
        v1_global_labels=semantic_v1_all,
        v1_original_labels=semantic_v1_original,
        original_to_global=original_to_global,
    )
    original_indices = projected["original_indices"]
    original_v1 = projected["original_v1_labels"]
    original_v2 = projected["original_hybrid_labels"]
    original_sources = projected["original_sources"]
    timers["fusion_and_original_mapping_seconds"] = time.perf_counter() - started

    config_identity = {
        "roi_bounds_half_open": [float(value) for value in args.roi],
        "voxel_size": args.voxel_size,
        "normal_angle_degrees": args.normal_angle,
        "min_region_points": args.min_region_points,
        "decision_min_score": args.decision_min_score,
        "input": str(args.input.resolve()),
    }
    namespace_hash = hashlib.sha256(json.dumps(config_identity, sort_keys=True).encode("utf-8"))
    namespace_hash.update(np.ascontiguousarray(roi_indices).view(np.uint8))
    namespace_hash.update(np.ascontiguousarray(local_regions).view(np.uint8))
    namespace = namespace_hash.hexdigest()[:12]
    region_instance_ids = np.asarray([
        f"hybrid-{namespace}:region-{record['region_id_local']}"
        for record in region_records
    ])
    for record, instance_id in zip(region_records, region_instance_ids):
        record["instance_namespace"] = instance_id

    started = time.perf_counter()
    output_dir.mkdir(parents=True)
    output_cloud = make_semantic_cloud(roi_points, roi_normals, hybrid_global)
    ply_path = output_dir / "hybrid_global_roi.ply"
    if not o3d.io.write_point_cloud(str(ply_path), output_cloud):
        raise OSError(f"No se pudo escribir {ply_path}")
    npz_path = output_dir / "hybrid_roi_labels.npz"
    np.savez_compressed(
        npz_path,
        roi_global_indices=roi_indices,
        roi_xyz=roi_points,
        roi_rgb_original=roi_rgb,
        roi_normals_original=roi_normals,
        geometry_labels_v1=roi_geometry,
        semantic_labels_v1_global=roi_v1,
        local_region_labels=local_regions,
        local_semantic_proposals=local_proposals,
        semantic_labels_v2_global=hybrid_global,
        source_global=source_global,
        original_indices=original_indices,
        semantic_labels_v1_original=original_v1,
        semantic_labels_v2_original=original_v2,
        source_original=original_sources,
        original_local_region_ids=local_regions[projected["local_global_rows"]],
        region_instance_ids=region_instance_ids,
        source_ids=np.asarray([SOURCE_IDS[name] for name in SOURCE_IDS], dtype=np.uint8),
        source_names=np.asarray(list(SOURCE_IDS)),
        class_ids=np.asarray(list(CLASS_IDS.values()), dtype=np.int8),
        class_names=np.asarray(list(CLASS_IDS)),
    )
    figure_path = output_dir / "hybrid_v1_v2_comparison.png"
    make_comparison_figure(figure_path, roi_points, roi_v1, hybrid_global, local_regions)
    timers["export_and_figures_seconds"] = time.perf_counter() - started

    unknown_without_segment = (roi_geometry < 0) & (roi_v1 == CLASS_IDS["unknown"])
    unknown_in_segment = (roi_geometry >= 0) & (roi_v1 == CLASS_IDS["unknown"])
    changes = original_v1 != original_v2
    report = {
        "method": "V1 is immutable; V2 applies local-region proposals only to V1-unknown ROI points.",
        "inputs": {
            "global_cloud": str(args.input.resolve()),
            "geometry_labels_v1": str(args.geometry_labels.resolve()),
            "semantic_v1": str(args.semantic_v1.resolve()),
            "mapping": str(args.mapping.resolve()),
            "planes_metrics": str(args.planes_metrics.resolve()),
        },
        "parameters": {
            **config_identity,
            "max_points": args.max_points,
            "max_edges": args.max_edges,
            "vertical_axis": args.vertical_axis,
            "vertical_axis_confirmed": args.vertical_confirmed,
            "classifier_rules": rule_config_dict(rule_config),
            "seed": args.seed,
        },
        "index_alignment": {
            "roi_global_points": int(len(roi_indices)),
            "roi_original_points": int(len(original_indices)),
            "original_mapping_out_of_range_rows": int(np.sum(~in_range)),
            "roi_global_rows_without_original_correspondence": int(np.sum(~mapping_valid_global)),
            "roi_original_rows_inconsistent_with_v1_global": projected["mapping_inconsistent_rows"],
            "region_indices_are_roi_local": True,
            "region_instance_namespace": f"hybrid-{namespace}",
        },
        "v1_unknown_provenance_roi_global": {
            "without_retained_geometric_segment": int(unknown_without_segment.sum()),
            "unknown_inside_retained_segment": int(unknown_in_segment.sum()),
            "total_unknown": int(np.sum(roi_v1 == CLASS_IDS["unknown"])),
        },
        "geometry_coverage": {
            "v1_percent": 100.0 * float(np.mean(roi_geometry >= 0)),
            "local_region_percent": 100.0 * float(np.mean(local_regions >= 0)),
            "local_region_noise_points": int(np.sum(local_regions < 0)),
        },
        "semantics_on_same_original_indices": {
            "v1_class_counts": class_counts(original_v1),
            "v2_class_counts": class_counts(original_v2),
            "v1_semantic_coverage_percent": percent(original_v1 != CLASS_IDS["unknown"]),
            "v2_semantic_coverage_percent": percent(original_v2 != CLASS_IDS["unknown"]),
            "v1_unknown_percent": percent(original_v1 == CLASS_IDS["unknown"]),
            "v2_unknown_percent": percent(original_v2 == CLASS_IDS["unknown"]),
            "changed_original_points": int(changes.sum()),
            "unchanged_original_points": int((~changes).sum()),
            "stability_percent": percent(~changes),
            "new_proposals_by_class": {
                name: int(np.sum((original_sources == SOURCE_IDS["local_region"]) & (original_v2 == class_id)))
                for name, class_id in CLASS_IDS.items()
            },
        },
        "fusion": {
            **fusion_metrics,
            "source_counts_original": {
                name: int(np.sum(original_sources == source_id))
                for name, source_id in SOURCE_IDS.items()
            },
            "known_v1_predictions_overwritten": 0,
            "inconsistent_original_rows_forced_unknown": projected["mapping_inconsistent_rows"],
        },
        "local_regions": {
            **region_metrics,
            "region_size_distribution": [record["point_count"] for record in region_records],
            "regions": region_records,
        },
        "evaluation": {
            "manual_ground_truth_available": False,
            "accuracy_reported": False,
            "iou_reported": False,
            "heuristic_scores_are_probabilities": False,
        },
        "timings_seconds": timers,
        "peak_working_set_bytes": None,
        "memory_measurement_note": "No fue posible leer el pico de working set desde esta sesión; los límites efectivos fueron max_points y max_edges.",
        "outputs": {
            "ply": str(ply_path),
            "labels_npz": str(npz_path),
            "comparison_figure": str(figure_path),
        },
        "output_ply_note": "Global-resolution ROI only; XYZ/normals preserved; RGB is replaced by fixed semantic colors.",
    }
    metrics_path = output_dir / "hybrid_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "roi_global_points": len(roi_indices),
        "roi_original_points": len(original_indices),
        "regions": region_metrics["regions"],
        "v1_semantic_coverage_percent": report["semantics_on_same_original_indices"]["v1_semantic_coverage_percent"],
        "v2_semantic_coverage_percent": report["semantics_on_same_original_indices"]["v2_semantic_coverage_percent"],
        "new_proposals_by_class": report["semantics_on_same_original_indices"]["new_proposals_by_class"],
        "changed_original_points": int(changes.sum()),
        "unknown_causes_roi": report["v1_unknown_provenance_roi_global"],
    }, indent=2))
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, MemoryError, KeyError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
