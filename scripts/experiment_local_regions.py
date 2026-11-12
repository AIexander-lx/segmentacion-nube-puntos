"""Compare bounded voxel regions with the existing clustering on giant-component crops."""
from __future__ import annotations

import argparse
import ctypes
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

from segmentacion.clustering import spatial_components
from segmentacion.geometric_graph import geometric_components
from segmentacion.io import load_ply
from segmentacion.local_regions import local_voxel_regions
from segmentacion.semantic_classifier import CLASS_IDS, classify_region, semantic_colors
from segmentacion.semantic_classifier import RuleConfig
from segmentacion.semantic_features import region_features


def peak_working_set_bytes() -> int | None:
    """Return this process' peak resident working set on Windows."""
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except (AttributeError, OSError):
        return None


def deterministic_region_colors(labels: np.ndarray) -> np.ndarray:
    """Map integer region IDs to stable colors; noise remains medium gray."""
    values = np.asarray(labels)
    colors = np.full((len(values), 3), 0.48, dtype=float)
    assigned = values >= 0
    if np.any(assigned):
        ids = values[assigned].astype(np.int64)
        hue = (ids * 0.61803398875) % 1.0
        # Compact HSV conversion in vector form, without another package.
        sector = np.floor(hue * 6).astype(int)
        fraction = hue * 6 - sector
        p = np.full(len(ids), 0.22)
        q = 0.95 * (1 - fraction * (1 - 0.22))
        t = 0.95 * (1 - (1 - fraction) * (1 - 0.22))
        rgb = np.empty((len(ids), 3))
        choices = (
            (0.95, t, 0.22), (q, 0.95, 0.22), (0.22, 0.95, t),
            (0.22, q, 0.95), (t, 0.22, 0.95), (0.95, 0.22, q),
        )
        for index, choice in enumerate(choices):
            select = sector % 6 == index
            if np.any(select):
                rgb[select] = np.column_stack([
                    np.full(select.sum(), value) if np.isscalar(value) else value[select]
                    for value in choice
                ])
        colors[assigned] = rgb
    return colors


def normal_coherence_by_label(labels: np.ndarray, normals: np.ndarray) -> dict[str, float | int]:
    """Measure within-label sign-invariant normal concentration."""
    valid = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-12)
    unit = np.zeros_like(normals, dtype=float)
    lengths = np.linalg.norm(normals[valid], axis=1)
    unit[valid] = normals[valid] / lengths[:, None]
    assigned_labels = np.unique(labels[labels >= 0])
    scores = []
    weights = []
    for label in assigned_labels:
        rows = (labels == label) & valid
        if np.sum(rows) < 3:
            continue
        sample = unit[rows]
        eigenvalues = np.linalg.eigvalsh(sample.T @ sample)
        scores.append(float(eigenvalues[-1] / len(sample)))
        weights.append(len(sample))
    return {
        "regions_with_normal_measure": len(scores),
        "mean_region_normal_coherence": float(np.mean(scores)) if scores else 0.0,
        "point_weighted_normal_coherence": float(np.average(scores, weights=weights)) if scores else 0.0,
    }


def choose_crops(
    giant_points: np.ndarray,
    source_metrics: dict,
    cell_size: float = 1.5,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Reuse saved dense/transition cells and find a high-aspect-ratio crop."""
    origin = giant_points.min(axis=0)
    keys = np.floor((giant_points - origin) / cell_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True,
    )
    cell_metrics = source_metrics["spatial_cells"]
    selected_keys = {
        "dense_transition": np.asarray(cell_metrics["dense"]["cell_key"], dtype=np.int64),
        "normal_transition": np.asarray(cell_metrics["normal_transition"]["cell_key"], dtype=np.int64),
    }

    cell_min = np.full((len(unique_keys), 3), np.inf)
    cell_max = np.full((len(unique_keys), 3), -np.inf)
    np.minimum.at(cell_min, inverse, giant_points)
    np.maximum.at(cell_max, inverse, giant_points)
    extents = cell_max - cell_min
    eligible = (counts >= 100) & (counts <= 3000)
    for known_key in selected_keys.values():
        eligible &= np.any(unique_keys != known_key, axis=1)
    aspect = extents.max(axis=1) / np.maximum(np.partition(extents, -2, axis=1)[:, -2], 1e-9)
    if not np.any(eligible):
        raise RuntimeError("No se encontraron recortes contiguos elegibles")
    elongated_id = np.flatnonzero(eligible)[np.argmax(aspect[eligible])]
    selected_keys["elongated_candidate"] = unique_keys[elongated_id]

    result = {}
    for name, key in selected_keys.items():
        cell_id = np.flatnonzero(np.all(unique_keys == key, axis=1))
        if not len(cell_id):
            raise ValueError(f"La celda {name} del diagnóstico no existe en el componente")
        cell_id = int(cell_id[0])
        local_rows = np.flatnonzero(inverse == cell_id)
        result[name] = (
            local_rows,
            {
                "cell_key": key.tolist(),
                "points": int(len(local_rows)),
                "extent_xyz": extents[cell_id].tolist(),
                "aspect_ratio": float(aspect[cell_id]),
                "crop_origin": origin.tolist(),
                "crop_cell_size": cell_size,
            },
        )
    return result


def run_method_comparisons(points: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> dict:
    """Run the saved clustering baseline, normal graph and three voxel scales."""
    comparisons = {}
    labels_by_method = {}
    started = time.perf_counter()
    baseline, baseline_metrics, baseline_seconds = spatial_components(
        points, eps=0.25, min_points=8, max_neighbors=64,
        min_cluster_size=10, max_cluster_size=20_000,
    )
    baseline_metrics["normal_summary"] = normal_coherence_by_label(baseline, normals)
    baseline_metrics["assigned_points"] = int(np.sum(baseline >= 0))
    comparisons["baseline_spatial"] = {**baseline_metrics, "seconds": baseline_seconds}
    labels_by_method["baseline_spatial"] = baseline

    graph, graph_metrics, graph_seconds = geometric_components(
        points, normals, eps=0.12, max_neighbors=24, normal_angle_degrees=35,
        min_component_size=10, max_edges=100_000,
    )
    graph_metrics["normal_summary"] = normal_coherence_by_label(graph, normals)
    graph_metrics["assigned_points"] = int(np.sum(graph >= 0))
    comparisons["normal_graph"] = {**graph_metrics, "seconds": graph_seconds}
    labels_by_method["normal_graph"] = graph

    local_labels_for_default = None
    local_metrics_for_default = None
    for voxel_size in (0.12, 0.20, 0.30):
        labels, metrics, elapsed = local_voxel_regions(
            points, normals,
            voxel_size=voxel_size,
            normal_angle_degrees=35,
            max_centroid_distance=2.0 * voxel_size,
            min_region_points=8,
            max_points=5_000,
            max_edges=100_000,
        )
        metrics["normal_summary"] = normal_coherence_by_label(labels, normals)
        metrics["assigned_points"] = int(np.sum(labels >= 0))
        method_name = f"voxel_regions_{voxel_size:.2f}"
        comparisons[method_name] = {**metrics, "seconds": elapsed}
        if abs(voxel_size - 0.20) < 1e-9:
            local_labels_for_default = labels
            local_metrics_for_default = metrics
        labels_by_method[method_name] = labels

    for angle in (20.0, 55.0):
        labels, metrics, elapsed = local_voxel_regions(
            points, normals,
            voxel_size=0.20,
            normal_angle_degrees=angle,
            max_centroid_distance=0.40,
            min_region_points=8,
            max_points=5_000,
            max_edges=100_000,
        )
        metrics["normal_summary"] = normal_coherence_by_label(labels, normals)
        metrics["assigned_points"] = int(np.sum(labels >= 0))
        method_name = f"voxel_regions_0.20_angle_{int(angle)}"
        comparisons[method_name] = {**metrics, "seconds": elapsed}
        labels_by_method[method_name] = labels

    return {
        "comparisons": comparisons,
        "labels_by_method": labels_by_method,
        "default_local_labels": local_labels_for_default,
        "default_local_metrics": local_metrics_for_default,
        "total_method_seconds": time.perf_counter() - started,
    }


def propose_local_semantics(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    region_labels: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    planes: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    semantic = np.full(len(points), CLASS_IDS["unknown"], dtype=np.int8)
    records = []
    # Keep the fixed 0.64 "other" fallback as unknown at this exploratory
    # scale; a small voxel patch can be mixed without being an "other object".
    local_rules = RuleConfig(decision_min_score=0.70)
    for region_id in np.unique(region_labels[region_labels >= 0]):
        rows = np.flatnonzero(region_labels == region_id)
        features = region_features(
            points[rows], normals=normals[rows], colors=colors[rows],
            scene_bounds=scene_bounds, vertical_axis=2, plane_models=planes,
        )
        prediction = classify_region(
            features, region_type="cluster", vertical_axis_confirmed=False,
            config=local_rules,
        )
        semantic[rows] = prediction["class_id"]
        records.append({
            "region_id": int(region_id),
            "point_count": int(len(rows)),
            "features": features,
            "prediction": prediction,
        })
    return semantic, records


def save_crop_mapping(
    output_dir: Path,
    crop_name: str,
    crop_global_indices: np.ndarray,
    region_labels: np.ndarray,
    original_to_voxel: np.ndarray,
    baseline_semantic: np.ndarray,
    baseline_geometry: np.ndarray,
    normal_graph_labels: np.ndarray,
    local_semantic: np.ndarray,
) -> Path:
    global_to_local = np.full(len(baseline_semantic), -1, dtype=np.int32)
    global_to_local[crop_global_indices] = np.arange(len(crop_global_indices), dtype=np.int32)
    original_mask = (original_to_voxel >= 0) & (global_to_local[np.maximum(original_to_voxel, 0)] >= 0)
    original_indices = np.flatnonzero(original_mask)
    local_rows = global_to_local[original_to_voxel[original_indices]]
    path = output_dir / f"{crop_name}_index_map.npz"
    np.savez_compressed(
        path,
        crop_global_indices=crop_global_indices,
        local_region_labels=region_labels,
        original_indices=original_indices,
        original_region_labels=region_labels[local_rows],
        semantic_v1_global=baseline_semantic[crop_global_indices],
        semantic_local_global=local_semantic,
        original_semantic_v1=baseline_semantic[crop_global_indices[local_rows]],
        original_semantic_local=local_semantic[local_rows],
        baseline_crop_cluster_labels=baseline_geometry,
        normal_graph_labels=normal_graph_labels,
    )
    return path


def draw_crop(
    output_dir: Path,
    crop_name: str,
    points: np.ndarray,
    rgb: np.ndarray,
    baseline_geometry: np.ndarray,
    normal_graph: np.ndarray,
    local_regions: np.ndarray,
    strict_local_regions: np.ndarray,
    semantic_v1: np.ndarray,
    local_semantic: np.ndarray,
) -> tuple[Path, Path]:
    region_figure = output_dir / f"{crop_name}_region_comparison.png"
    semantic_figure = output_dir / f"{crop_name}_semantic_comparison.png"
    panels = (
        (rgb, "RGB original"),
        (deterministic_region_colors(baseline_geometry), "Baseline espacial"),
        (deterministic_region_colors(normal_graph), "Grafo con normales"),
        (deterministic_region_colors(local_regions), "Regiones voxel locales"),
        (deterministic_region_colors(strict_local_regions), "Voxel local: ángulo 20°"),
    )
    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    for axis, (colors, title) in zip(axes.flat[:5], panels):
        axis.scatter(points[:, 0], points[:, 2], c=colors, s=4, linewidths=0, rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_aspect("equal", adjustable="datalim")
    semantic_panels = (
        (semantic_colors(semantic_v1), "Semántica V1"),
        (semantic_colors(local_semantic), "Semántica local exploratoria"),
    )
    for axis, (colors, title) in zip(axes.flat[5:7], semantic_panels):
        axis.scatter(points[:, 0], points[:, 2], c=colors, s=4, linewidths=0, rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_aspect("equal", adjustable="datalim")
    for axis in axes.flat[7:]:
        axis.set_visible(False)
    figure.suptitle(f"{crop_name}: regiones no equivalen a objetos semánticos")
    figure.tight_layout()
    figure.savefig(region_figure, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, labels, title in (
        (axes[0], semantic_v1, "Baseline V1"),
        (axes[1], local_semantic, "Propuestas locales"),
    ):
        axis.scatter(points[:, 0], points[:, 2], c=semantic_colors(labels), s=4, linewidths=0)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle(f"{crop_name}: unknown se conserva; sin etiquetas de referencia")
    figure.tight_layout()
    figure.savefig(semantic_figure, dpi=180)
    plt.close(figure)
    return region_figure, semantic_figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/local_regions_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescribe el resultado existente: {output_dir}")

    global_cloud = load_ply(ROOT / "outputs/processed/global.ply")
    points = np.asarray(global_cloud.points)
    normals = np.asarray(global_cloud.normals)
    colors = np.asarray(global_cloud.colors)
    with np.load(ROOT / "outputs/experiments/giant_component_v3/giant_indices.npz") as saved:
        giant_global_indices = saved["global_indices"].copy()
    with np.load(ROOT / "outputs/segmented/global_review_final/labels.npz") as saved:
        baseline_geometry_all = saved["global_labels"].copy()
    with np.load(ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz") as saved:
        baseline_semantic_all = saved["semantic_labels_global"].copy()
        baseline_semantic_original = saved["semantic_labels_original"].copy()
    with np.load(ROOT / "outputs/processed/global_mapping.npz") as saved:
        original_to_voxel = saved["original_to_voxel"].copy()
    source_metrics = json.loads(
        (ROOT / "outputs/experiments/giant_component_v3/metrics.json").read_text(encoding="utf-8")
    )
    segmentation_metrics = json.loads(
        (ROOT / "outputs/segmented/global_review_final/metrics.json").read_text(encoding="utf-8")
    )
    if len(baseline_geometry_all) != len(points) or len(baseline_semantic_all) != len(points):
        raise ValueError("Las etiquetas baseline no están alineadas con global.ply")
    if len(np.unique(giant_global_indices)) != len(giant_global_indices):
        raise ValueError("Los índices de la componente gigante tienen duplicados")
    if np.any(giant_global_indices < 0) or np.any(giant_global_indices >= len(points)):
        raise ValueError("Índice de componente gigante fuera de global.ply")
    if np.any(original_to_voxel >= len(points)):
        raise ValueError("original_to_voxel contiene índices fuera de global.ply")
    if len(baseline_semantic_original) != len(original_to_voxel):
        raise ValueError("Etiquetas semánticas originales no coinciden con el mapeo")
    valid_original_mapping = original_to_voxel >= 0
    original_geometry_labels = np.full(len(original_to_voxel), -1, dtype=np.int32)
    original_geometry_labels[valid_original_mapping] = baseline_geometry_all[original_to_voxel[valid_original_mapping]]

    giant_points = points[giant_global_indices]
    crops = choose_crops(giant_points, source_metrics)
    output_dir.mkdir(parents=True)
    report = {
        "method_note": "Las regiones voxel son unidades geométricas locales, no objetos ni clases.",
        "inputs": {
            "global_cloud": str((ROOT / "outputs/processed/global.ply").resolve()),
            "giant_component_indices": str((ROOT / "outputs/experiments/giant_component_v3/giant_indices.npz").resolve()),
            "semantic_v1": str((ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz").resolve()),
            "original_to_voxel": str((ROOT / "outputs/processed/global_mapping.npz").resolve()),
        },
        "parameters": {
            "seed": args.seed,
            "crop_cell_size": 1.5,
            "voxel_sizes_tested": [0.12, 0.20, 0.30],
            "normal_angle_degrees": 35,
            "local_max_points": 5_000,
            "local_max_edges": 100_000,
            "baseline": {"eps": 0.25, "min_points": 8, "max_neighbors": 64},
            "normal_graph": {"eps": 0.12, "max_neighbors": 24, "normal_angle_degrees": 35},
            "local_semantic_rules": {"decision_min_score": 0.70, "other_class_fallback_suppressed": True},
        },
        "giant_component": {
            "global_points": int(len(giant_global_indices)),
            "baseline_geometry_labels": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(baseline_geometry_all[giant_global_indices], return_counts=True))
            },
            "baseline_semantic_classes": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(baseline_semantic_all[giant_global_indices], return_counts=True))
            },
        },
        "v1_unknown_provenance": {
            "global": {
                "without_retained_geometric_segment": int(np.sum(baseline_geometry_all < 0)),
                "unknown_inside_retained_segment": int(np.sum((baseline_geometry_all >= 0) & (baseline_semantic_all == CLASS_IDS["unknown"]))),
                "total_semantic_unknown": int(np.sum(baseline_semantic_all == CLASS_IDS["unknown"])),
            },
            "original": {
                "without_retained_geometric_segment": int(np.sum(original_geometry_labels < 0)),
                "unknown_inside_retained_segment": int(np.sum((original_geometry_labels >= 0) & (baseline_semantic_original == CLASS_IDS["unknown"]))),
                "total_semantic_unknown": int(np.sum(baseline_semantic_original == CLASS_IDS["unknown"])),
            },
            "note": "Noise and oversized rejected components both appear without a retained geometric segment.",
        },
        "crops": {},
        "outputs": {},
        "evaluation": {"ground_truth_available": False, "accuracy_reported": False, "iou_reported": False},
    }

    for crop_name, (giant_local_indices, crop_info) in crops.items():
        global_indices = giant_global_indices[giant_local_indices]
        crop_points = points[global_indices]
        crop_normals = normals[global_indices]
        crop_colors = colors[global_indices]
        compared = run_method_comparisons(crop_points, crop_normals, crop_colors)
        method_labels = compared["labels_by_method"]
        local_regions = compared["default_local_labels"]
        crop_scene = (points.min(axis=0), points.max(axis=0))
        local_semantic, local_semantic_records = propose_local_semantics(
            crop_points, crop_normals, crop_colors, local_regions,
            crop_scene, segmentation_metrics.get("planes", []),
        )
        region_sizes = np.bincount(local_regions[local_regions >= 0])
        semantic_coverage = int(np.sum(local_semantic != CLASS_IDS["unknown"]))
        baseline_assigned_semantic = int(np.sum(baseline_semantic_all[global_indices] != CLASS_IDS["unknown"]))
        indices_path = save_crop_mapping(
            output_dir, crop_name, global_indices, local_regions,
            original_to_voxel, baseline_semantic_all, method_labels["baseline_spatial"],
            method_labels["normal_graph"], local_semantic,
        )
        region_figure, semantic_figure = draw_crop(
            output_dir, crop_name, crop_points, crop_colors,
            method_labels["baseline_spatial"], method_labels["normal_graph"],
            local_regions, method_labels["voxel_regions_0.20_angle_20"],
            baseline_semantic_all[global_indices], local_semantic,
        )
        report["crops"][crop_name] = {
            **crop_info,
            "global_indices_preserved": True,
            "original_point_count_in_crop": int(np.sum(np.isin(original_to_voxel, global_indices))),
            "methods": compared["comparisons"],
            "selected_local_regions": {
                "voxel_size": 0.20,
                "region_count": int(len(region_sizes)),
                "region_size_distribution": {
                    "min": int(region_sizes.min()) if len(region_sizes) else 0,
                    "median": float(np.median(region_sizes)) if len(region_sizes) else 0.0,
                    "max": int(region_sizes.max()) if len(region_sizes) else 0,
                    "sizes": region_sizes.tolist(),
                },
                "geometric_coverage_percent": 100.0 * float(np.mean(local_regions >= 0)),
                "noise_points": int(np.sum(local_regions < 0)),
                "semantic_coverage_percent": 100.0 * semantic_coverage / max(len(local_semantic), 1),
                "semantic_class_counts": {
                    name: int(np.sum(local_semantic == class_id))
                    for name, class_id in CLASS_IDS.items()
                },
                "semantic_regions": local_semantic_records,
            },
            "baseline_v1_same_crop": {
                "geometric_coverage_percent": 100.0 * float(np.mean(baseline_geometry_all[global_indices] >= 0)),
                "semantic_coverage_percent": 100.0 * baseline_assigned_semantic / len(global_indices),
                "geometry_unassigned_points": int(np.sum(baseline_geometry_all[global_indices] < 0)),
                "semantic_unknown_points": int(np.sum(baseline_semantic_all[global_indices] == CLASS_IDS["unknown"])),
            },
            "outputs": {
                "indices_npz": str(indices_path),
                "region_figure": str(region_figure),
                "semantic_figure": str(semantic_figure),
            },
        }
        report["outputs"][crop_name] = report["crops"][crop_name]["outputs"]

    report_path = output_dir / "local_regions_metrics.json"
    report["peak_working_set_bytes"] = peak_working_set_bytes()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        name: {
            "points": info["points"],
            "cell_key": info["cell_key"],
            "methods": {
                method: {
                    "regions": values.get("clusters", values.get("components", values.get("regions"))),
                    "assigned": values["assigned_points"],
                    "seconds": values["seconds"],
                    "normal": values.get("normal_summary", {}).get("mean_region_normal_coherence"),
                }
                for method, values in info["methods"].items()
            },
            "local_semantic_coverage_percent": info["selected_local_regions"]["semantic_coverage_percent"],
        }
        for name, info in report["crops"].items()
    }, indent=2))
    print(f"Metrics: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, MemoryError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
