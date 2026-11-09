"""Create geometric semantic proposals for an existing global segmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from segmentacion.io import load_ply
from segmentacion.semantic_classifier import (
    CLASS_COLORS,
    CLASS_IDS,
    RuleConfig,
    classify_region,
    rule_config_dict,
    semantic_colors,
)
from segmentacion.semantic_features import region_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "FARO_Scan_Full_Sub1_down3_with_normals.ply")
    parser.add_argument("--global-cloud", type=Path, default=ROOT / "outputs/processed/global.ply")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/processed/global_mapping.npz")
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs/segmented/global_review_final/labels.npz")
    parser.add_argument("--planes-metrics", type=Path, default=ROOT / "outputs/segmented/global_review_final/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/semantic/run_001")
    parser.add_argument("--seed", type=int, default=42, help="Seed de la figura reproducible")
    parser.add_argument("--vertical-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--vertical-confirmed", action="store_true", help="El eje se confirmó con referencia externa")
    parser.add_argument("--decision-min-score", type=float, default=0.62)
    parser.add_argument("--decision-margin", type=float, default=0.12)
    parser.add_argument("--linearity-min", type=float, default=0.72)
    parser.add_argument("--normal-radial-min", type=float, default=0.65)
    parser.add_argument("--vertical-alignment-min", type=float, default=0.80)
    parser.add_argument("--plane-shape-min", type=float, default=0.55)
    parser.add_argument("--cross-section-ratio-max", type=float, default=2.2)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    original = load_ply(args.original)
    global_cloud = load_ply(args.global_cloud)
    points = np.asarray(global_cloud.points)
    normals = np.asarray(global_cloud.normals) if global_cloud.has_normals() else None
    colors = np.asarray(global_cloud.colors) if global_cloud.has_colors() else None
    with np.load(args.labels) as saved_labels:
        labels = saved_labels["global_labels"].copy()
    with np.load(args.mapping) as saved_mapping:
        mapping = saved_mapping["original_to_voxel"].copy()

    if len(labels) != len(points) or len(mapping) != len(original.points):
        raise ValueError("Las etiquetas o el mapeo no coinciden con las nubes de entrada")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("original_to_voxel debe contener índices enteros")
    valid = mapping >= 0
    if np.any(mapping[valid] >= len(labels)) or len(np.unique(mapping[valid])) != len(points):
        raise ValueError("Mapeo original→global inválido o incompleto")
    return original, global_cloud, labels, mapping, points


def segment_groups(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return stable segment IDs and the corresponding row indices."""
    assigned_indices = np.flatnonzero(labels >= 0)
    if not len(assigned_indices):
        return []
    order = np.argsort(labels[assigned_indices], kind="stable")
    ordered_indices = assigned_indices[order]
    ordered_labels = labels[ordered_indices]
    boundaries = np.flatnonzero(np.diff(ordered_labels)) + 1
    chunks = np.split(ordered_indices, boundaries)
    return [(int(labels[chunk[0]]), chunk) for chunk in chunks]


def build_proposals(
    points: np.ndarray,
    global_cloud: Any,
    labels: np.ndarray,
    plane_models: dict[int, dict[str, Any]],
    vertical_axis: int,
    vertical_confirmed: bool,
    config: RuleConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    normals = np.asarray(global_cloud.normals) if global_cloud.has_normals() else None
    colors = np.asarray(global_cloud.colors) if global_cloud.has_colors() else None
    scene_bounds = (np.percentile(points, 0.1, axis=0), np.percentile(points, 99.9, axis=0))  # robusto a puntos aislados
    semantic = np.full(len(points), CLASS_IDS["unknown"], dtype=np.int8)
    records = []

    for segment_id, indices in segment_groups(labels):
        plane = plane_models.get(segment_id)
        segment_type = "plane" if plane is not None else "cluster"
        features = region_features(
            points[indices],
            normals=normals[indices] if normals is not None else None,
            colors=colors[indices] if colors is not None else None,
            scene_bounds=scene_bounds,
            vertical_axis=vertical_axis,
            plane_models=list(plane_models.values()),
        )
        prediction = classify_region(
            features,
            region_type=segment_type,
            plane_model=plane,
            vertical_axis_confirmed=vertical_confirmed,
            config=config,
        )
        semantic[indices] = prediction["class_id"]
        records.append({
            "segment_id": segment_id,
            "segment_type": segment_type,
            "point_count_global": int(len(indices)),
            "features": features,
            "prediction": prediction,
        })
    return semantic, records


def count_classes(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    id_to_name = {value: name for name, value in CLASS_IDS.items()}
    result = {name: 0 for name in CLASS_IDS}
    result.update({id_to_name[int(value)]: int(count) for value, count in zip(values, counts)})
    return result


def write_figures(
    output_dir: Path,
    original_points: np.ndarray,
    original_labels: np.ndarray,
    seed: int,
) -> tuple[Path, Path]:
    counts = count_classes(original_labels)
    distribution_path = output_dir / "class_distribution.png"
    figure, axis = plt.subplots(figsize=(10, 4))
    names = list(CLASS_IDS)
    axis.bar(names, [counts[name] for name in names], color=[CLASS_COLORS[name] for name in names])
    axis.set_ylabel("Puntos originales")
    axis.set_title("Propuestas heurísticas (no ground truth)")
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=180)
    plt.close(figure)

    projection_path = output_dir / "semantic_projections.png"
    generator = np.random.default_rng(seed)
    sample = generator.choice(len(original_points), min(100_000, len(original_points)), replace=False)
    rgb = semantic_colors(original_labels[sample])
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (first, second, title) in zip(axes, ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))):
        axis.scatter(
            original_points[sample, first], original_points[sample, second],
            c=rgb, s=0.15, linewidths=0, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("XYZ"[first])
        axis.set_ylabel("XYZ"[second])
        axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(projection_path, dpi=180)
    plt.close(figure)
    return distribution_path, projection_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescribe resultado existente: {output_dir}")
    timers: dict[str, float] = {}

    started = perf_counter()
    original, global_cloud, labels, mapping, global_points = load_inputs(args)
    timers["load_and_validate_seconds"] = perf_counter() - started

    started = perf_counter()
    with args.planes_metrics.open(encoding="utf-8") as file:
        baseline_metrics = json.load(file)
    planes = {int(model["id"]): model for model in baseline_metrics.get("planes", [])}
    axis_index = "xyz".index(args.vertical_axis)
    config = RuleConfig(
        vertical_alignment_min=args.vertical_alignment_min,
        plane_shape_min=args.plane_shape_min,
        decision_min_score=args.decision_min_score,
        decision_margin=args.decision_margin,
        linearity_min=args.linearity_min,
        normal_radial_min=args.normal_radial_min,
        cross_section_ratio_max=args.cross_section_ratio_max,
    )
    global_semantic, segments = build_proposals(
        global_points, global_cloud, labels, planes, axis_index, args.vertical_confirmed, config,
    )
    timers["features_and_rules_seconds"] = perf_counter() - started

    started = perf_counter()
    mapping_valid = mapping >= 0
    original_semantic = np.full(len(mapping), CLASS_IDS["unknown"], dtype=np.int8)
    original_semantic[mapping_valid] = global_semantic[mapping[mapping_valid]]
    timers["propagation_seconds"] = perf_counter() - started

    started = perf_counter()
    output_dir.mkdir(parents=True)
    original_cloud = o3d.geometry.PointCloud(original)
    original_cloud.colors = o3d.utility.Vector3dVector(semantic_colors(original_semantic))
    ply_path = output_dir / "semantic_original.ply"
    if not o3d.io.write_point_cloud(str(ply_path), original_cloud):
        raise OSError(f"No se pudo exportar {ply_path}")
    npz_path = output_dir / "semantic_labels.npz"
    np.savez_compressed(
        npz_path,
        semantic_labels_global=global_semantic,
        semantic_labels_original=original_semantic,
        class_ids=np.asarray(list(CLASS_IDS.values()), dtype=np.int8),
        class_names=np.asarray(list(CLASS_IDS)),
        class_colors=np.asarray([CLASS_COLORS[name] for name in CLASS_IDS], dtype=float),
    )
    distribution_figure, projection_figure = write_figures(
        output_dir, np.asarray(original.points), original_semantic, args.seed,
    )
    timers["export_and_figures_seconds"] = perf_counter() - started

    global_counts = count_classes(global_semantic)
    original_counts = count_classes(original_semantic)
    report = {
        "source": {
            "original": str(args.original.resolve()),
            "global_cloud": str(args.global_cloud.resolve()),
            "segment_labels": str(args.labels.resolve()),
            "mapping": str(args.mapping.resolve()),
            "plane_metrics": str(args.planes_metrics.resolve()),
        },
        "index_alignment": {
            "original_points": len(original.points),
            "global_points": len(global_points),
            "mapped_original_points": int(mapping_valid.sum()),
            "invalid_mapping_points": int((~mapping_valid).sum()),
        },
        "method": "interpretable geometric heuristics; not trained and not ground truth",
        "parameters": {
            "seed": args.seed,
            "vertical_axis": args.vertical_axis,
            "vertical_axis_confirmed": args.vertical_confirmed,
            "rule_config": rule_config_dict(config),
        },
        "classes": {"fixed_ids": CLASS_IDS, "fixed_rgb": CLASS_COLORS},
        "coverage": {
            "global": {
                "assigned_semantic": int(np.sum(global_semantic != CLASS_IDS["unknown"])),
                "unknown": int(np.sum(global_semantic == CLASS_IDS["unknown"])),
                "unknown_percent": 100.0 * float(np.mean(global_semantic == CLASS_IDS["unknown"])),
                "counts": global_counts,
            },
            "original": {
                "assigned_semantic": int(np.sum(original_semantic != CLASS_IDS["unknown"])),
                "unknown": int(np.sum(original_semantic == CLASS_IDS["unknown"])),
                "unknown_percent": 100.0 * float(np.mean(original_semantic == CLASS_IDS["unknown"])),
                "counts": original_counts,
            },
        },
        "segments": segments,
        "evaluation": {
            "ground_truth_available": False,
            "accuracy_reported": False,
            "semantic_iou_reported": False,
            "predictions_are_not_manual_annotations": True,
        },
        "timings_seconds": timers,
        "outputs": {
            "ply": str(ply_path),
            "labels_npz": str(npz_path),
            "class_distribution_figure": str(distribution_figure),
            "projection_figure": str(projection_figure),
        },
    }
    metrics_path = output_dir / "semantic_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["coverage"], indent=2, ensure_ascii=False))
    print(f"PLY: {ply_path}")
    print(f"Report: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
