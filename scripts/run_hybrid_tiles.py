"""Run hybrid V2 on bounded ROI(s) or an explicitly enabled tiled global pass."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from itertools import product
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
    REJECTION_REASON_BITS,
    build_spatial_tile_plan,
    compare_tile_proposals,
    finalize_tiled_predictions,
    make_semantic_cloud,
    preflight_tile_plan,
    project_roi_to_original,
    proposal_rejection_flags,
    select_roi_indices,
    tile_context_indices,
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--roi", type=float, nargs=6, action="append", metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    mode.add_argument("--global-experiment", action="store_true")
    parser.add_argument("--allow-global", action="store_true", help="Consentimiento explícito adicional para modo global")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/processed/global.ply")
    parser.add_argument("--geometry-labels", type=Path, default=ROOT / "outputs/segmented/global_review_final/labels.npz")
    parser.add_argument("--semantic-v1", type=Path, default=ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/processed/global_mapping.npz")
    parser.add_argument("--planes-metrics", type=Path, default=ROOT / "outputs/segmented/global_review_final/metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=float, default=5.0)
    parser.add_argument("--halo", type=float, default=0.5)
    parser.add_argument("--reconciliation", choices=("baseline", "consensus"), default="baseline")
    parser.add_argument("--min-known-votes", type=int, default=2)
    parser.add_argument("--origin-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--normal-angle", type=float, default=35.0)
    parser.add_argument("--min-region-points", type=int, default=8)
    parser.add_argument("--decision-min-score", type=float, default=0.70)
    parser.add_argument("--max-tile-points", type=int, default=20_000)
    parser.add_argument("--max-edges", type=int, default=500_000)
    parser.add_argument("--max-total-context-points", type=int, default=1_000_000)
    parser.add_argument("--max-memory-mb", type=int, default=1024)
    parser.add_argument("--export-reserve-mb", type=int, default=128)
    parser.add_argument("--max-tiles", type=int, default=2_000)
    parser.add_argument("--max-roi-points", type=int, default=20_000)
    parser.add_argument("--max-global-points", type=int, default=400_000)
    parser.add_argument("--max-original-points", type=int, default=2_000_000)
    parser.add_argument("--vertical-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--vertical-confirmed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_memory_snapshot() -> tuple[int, int] | None:
    """Return current and peak resident process memory on Windows."""
    if sys.platform != "win32":
        return None

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(MemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
        return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)) if ok else None
    except (AttributeError, OSError):
        return None


def peak_working_set_bytes() -> int | None:
    snapshot = process_memory_snapshot()
    return snapshot[1] if snapshot is not None else None


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(labels == class_id)) for name, class_id in CLASS_IDS.items()}


def region_size_summary(tile_ids: np.ndarray, region_ids: np.ndarray) -> dict[str, float | int]:
    """Count owner-region sizes by (tile, local ID), never by local ID alone."""
    tiles = np.asarray(tile_ids)
    regions = np.asarray(region_ids)
    if tiles.shape != regions.shape or tiles.ndim != 1:
        raise ValueError("tile_ids y region_ids deben ser vectores alineados")
    valid = (tiles >= 0) & (regions >= 0)
    if not np.any(valid):
        return {"region_count": 0, "min_points": 0, "median_points": 0.0, "max_points": 0}
    pairs = np.column_stack((tiles[valid], regions[valid]))
    _, sizes = np.unique(pairs, axis=0, return_counts=True)
    return {
        "region_count": int(len(sizes)),
        "min_points": int(sizes.min()),
        "median_points": float(np.median(sizes)),
        "max_points": int(sizes.max()),
    }


def rejection_flag_counts(flags: np.ndarray) -> dict[str, int]:
    values = np.asarray(flags, dtype=np.uint8)
    return {
        name: int(np.sum((values & bit) != 0))
        for name, bit in REJECTION_REASON_BITS.items()
    }


def choose_target_indices(
    points: np.ndarray,
    rois: list[list[float]] | None,
    max_roi_points: int,
) -> np.ndarray:
    if rois is None:
        return np.arange(len(points), dtype=np.int64)
    selected = []
    for values in rois:
        bounds = (np.asarray(values[::2]), np.asarray(values[1::2]))
        selected.append(select_roi_indices(points, bounds, max_points=max_roi_points))
    target = np.unique(np.concatenate(selected))
    if len(target) > max_roi_points:
        raise MemoryError(f"Los ROI juntos suman {len(target)} puntos; límite max_roi_points={max_roi_points}")
    return target


def tiles_touching_targets(plan, points: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Find owner and halo tiles whose context contains at least one target row."""
    target_sorted = np.sort(target_indices)
    owner_ids = np.unique(plan.point_tile_ids[target_indices])
    candidates = set(int(value) for value in owner_ids)
    for owner_id in owner_ids:
        key = plan.tile_keys[int(owner_id)]
        for offset in product((-1, 0, 1), repeat=3):
            neighbor_key = tuple(int(key[axis] + offset[axis]) for axis in range(3))
            neighbor_id = plan.key_to_id.get(neighbor_key)
            if neighbor_id is None:
                continue
            context = tile_context_indices(plan, points, neighbor_id)
            if np.intersect1d(context, target_sorted, assume_unique=True).size:
                candidates.add(neighbor_id)
    return np.asarray(sorted(candidates), dtype=np.int64)


def _tile_namespace(configuration: dict, key: np.ndarray) -> str:
    payload = {**configuration, "tile_key": [int(value) for value in key]}
    return "tile-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def predict_context(
    context_indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float,
    normal_angle: float,
    min_region_points: int,
    max_tile_points: int,
    max_edges: int,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    plane_models: list[dict],
    vertical_axis: int,
    vertical_confirmed: bool,
    rules: RuleConfig,
) -> tuple[np.ndarray, np.ndarray, dict, float, list[dict]]:
    """Segment and classify only one tile's bounded context."""
    context_points = points[context_indices]
    context_normals = normals[context_indices]
    context_colors = colors[context_indices]
    regions, metrics, elapsed = local_voxel_regions(
        context_points,
        context_normals,
        voxel_size=voxel_size,
        normal_angle_degrees=normal_angle,
        min_region_points=min_region_points,
        max_points=max_tile_points,
        max_edges=max_edges,
    )
    proposals = np.full(len(context_indices), CLASS_IDS["unknown"], dtype=np.int8)
    summaries = []
    for region_id in np.unique(regions[regions >= 0]):
        rows = np.flatnonzero(regions == region_id)
        features = region_features(
            context_points[rows],
            normals=context_normals[rows],
            colors=context_colors[rows],
            scene_bounds=scene_bounds,
            vertical_axis=vertical_axis,
            plane_models=plane_models,
        )
        prediction = classify_region(
            features,
            region_type="cluster",
            vertical_axis_confirmed=vertical_confirmed,
            config=rules,
        )
        proposals[rows] = prediction["class_id"]
        summaries.append({
            "region_id_local": int(region_id),
            "point_count": int(len(rows)),
            "class": prediction["class"],
            "score_heuristic": float(prediction["score"]),
            "evidence": prediction.get("evidence", []),
            "features_valid": bool(features.get("valid", False)),
            "geometry": {key: features.get(key) for key in ("extent_xyz", "principal_axis", "linearity", "planarity", "curvature_approx", "radial_normal_fraction")},
        })
    return regions, proposals, metrics, elapsed, summaries


def draw_tile_figure(
    path: Path,
    points: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    tile_ids: np.ndarray,
    owner_regions: np.ndarray,
    disagreements: np.ndarray,
    seed: int,
) -> None:
    sample_count = min(100_000, len(points))
    if len(points) > sample_count:
        sample = np.random.default_rng(seed).choice(len(points), sample_count, replace=False)
    else:
        sample = np.arange(len(points))
    region_key = np.maximum(tile_ids[sample], 0).astype(np.int64) * 1_000_000 + np.maximum(owner_regions[sample], 0)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = (
        (semantic_colors(v1[sample]), "V1"),
        (semantic_colors(v2[sample]), "V2 tile"),
        (plt.get_cmap("turbo")((region_key * 0.61803398875) % 1.0)[:, :3], "Tile + región (ID local)"),
        (np.where(disagreements[sample, None], np.array([0.95, 0.05, 0.05]), np.array([0.45, 0.45, 0.45])), "Discrepancias de halo"),
    )
    for axis, (colors, title) in zip(axes, panels):
        axis.scatter(points[sample, 0], points[sample, 2], c=colors, s=1.2, linewidths=0, rasterized=True)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle("V1/V2 en los mismos índices; rojo = discrepancia owner/contexto")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def context_disagreement_mask(votes: np.ndarray, owner_labels: np.ndarray) -> np.ndarray:
    """Flag rows where at least one context tile disagrees with the owner."""
    vote_values = np.asarray(votes)
    labels = np.asarray(owner_labels)
    if vote_values.ndim != 2 or vote_values.shape[0] != len(labels):
        raise ValueError("Los votos deben alinearse con las etiquetas del propietario")
    return compare_tile_proposals(
        labels, vote_values, np.ones(len(labels), dtype=np.uint8),
    )["disagreement_mask"]


def draw_roi_figures(
    output_dir: Path,
    points: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    tile_ids: np.ndarray,
    owner_regions: np.ndarray,
    disagreements: np.ndarray,
    rois: list[list[float]],
    tile_size: float,
    origin: np.ndarray,
    seed: int,
) -> list[str]:
    """Write a zoomed, deterministic comparison for each contiguous ROI."""
    outputs = []
    for roi_number, bounds in enumerate(rois, start=1):
        lower = np.asarray(bounds[::2], dtype=float)
        upper = np.asarray(bounds[1::2], dtype=float)
        rows = np.flatnonzero(np.all((points >= lower) & (points < upper), axis=1))
        if not len(rows):
            continue
        rng = np.random.default_rng(seed + roi_number)
        sample = rows if len(rows) <= 100_000 else np.sort(rng.choice(rows, 100_000, replace=False))
        region_key = np.maximum(tile_ids[sample], 0).astype(np.int64) * 1_000_000 + np.maximum(owner_regions[sample], 0)
        panels = (
            (semantic_colors(v1[sample]), "V1"),
            (semantic_colors(v2[sample]), "V2 final"),
            (plt.get_cmap("turbo")((region_key * 0.61803398875) % 1.0)[:, :3], "Tile + región"),
            (np.where(disagreements[sample, None], [0.95, 0.05, 0.05], [0.45, 0.45, 0.45]), "Desacuerdo de contexto"),
        )
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for axis, (colors, title) in zip(axes, panels):
            axis.scatter(points[sample, 0], points[sample, 2], c=colors, s=2, linewidths=0, rasterized=True)
            axis.set_title(title)
            axis.set_xlabel("X")
            axis.set_ylabel("Z")
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[2], upper[2])
            for dimension, axis_min, axis_max, draw_line in (
                (0, lower[0], upper[0], axis.axvline),
                (2, lower[2], upper[2], axis.axhline),
            ):
                first = int(np.ceil((axis_min - origin[dimension]) / tile_size))
                last = int(np.floor((axis_max - origin[dimension]) / tile_size))
                for boundary_index in range(first, last + 1):
                    draw_line(origin[dimension] + boundary_index * tile_size,
                              color="black", alpha=0.35, linewidth=0.65, linestyle="--")
        fig.suptitle(f"ROI {roi_number}: mismo índice global; rojo = conflicto owner/contexto")
        fig.tight_layout()
        path = output_dir / f"roi_{roi_number:02d}_tile_comparison.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescriben resultados: {output_dir}")
    if args.global_experiment and not args.allow_global:
        raise ValueError("Use --allow-global junto con --global-experiment tras revisar el preflight.")
    if args.max_tile_points > 20_000:
        raise ValueError("max_tile_points no puede superar 20,000 en esta versión experimental")
    if args.max_edges > 500_000:
        raise ValueError("max_edges no puede superar 500,000 en esta versión experimental")
    if args.global_experiment and args.max_global_points > 400_000:
        raise ValueError("max_global_points excede el límite experimental de 400,000")

    if args.max_tiles > 5_000:
        raise ValueError("max_tiles no puede superar 5,000 en esta versiÃ³n experimental")
    if args.max_total_context_points > 2_000_000:
        raise ValueError("max_total_context_points no puede superar 2,000,000")
    if args.max_memory_mb < 256 or args.max_memory_mb > 4096:
        raise ValueError("max_memory_mb debe estar entre 256 y 4,096 MB")
    if args.export_reserve_mb < 64 or args.export_reserve_mb > 512:
        raise ValueError("export_reserve_mb debe estar entre 64 y 512 MB")
    if args.max_roi_points > 100_000:
        raise ValueError("max_roi_points no puede superar 100,000")
    if args.max_original_points > 2_000_000:
        raise ValueError("max_original_points no puede superar 2,000,000")

    if args.min_known_votes < 2:
        raise ValueError("min-known-votes debe ser al menos 2")
    if not np.isfinite(args.origin_offset).all():
        raise ValueError("origin-offset debe ser finito")
    timers: dict[str, float] = {}
    started = time.perf_counter()
    cloud = load_ply(args.input)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals) if cloud.has_normals() else np.zeros_like(points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.zeros_like(points)
    with np.load(args.geometry_labels) as saved:
        geometry_labels = saved["global_labels"].copy()
    with np.load(args.semantic_v1) as saved:
        semantic_v1 = saved["semantic_labels_global"].copy()
        semantic_v1_original = saved["semantic_labels_original"].copy()
    with np.load(args.mapping) as saved:
        original_to_global = saved["original_to_voxel"].copy()
    planes = json.loads(args.planes_metrics.read_text(encoding="utf-8")).get("planes", [])
    timers["load_inputs_seconds"] = time.perf_counter() - started

    if len(points) > args.max_global_points:
        raise MemoryError(f"La nube global tiene {len(points)} puntos; límite {args.max_global_points}")
    if len(geometry_labels) != len(points) or len(semantic_v1) != len(points):
        raise ValueError("Las etiquetas geométricas/semánticas V1 no coinciden con la nube global")
    if not np.issubdtype(original_to_global.dtype, np.integer):
        raise ValueError("original_to_voxel debe ser entero")
    if len(original_to_global) != len(semantic_v1_original):
        raise ValueError("El mapeo y las etiquetas V1 originales no están alineados")
    if args.global_experiment and len(original_to_global) > args.max_original_points:
        raise MemoryError("La cantidad de puntos originales excede max_original_points")

    valid_mapping = (original_to_global >= 0) & (original_to_global < len(points))
    if args.global_experiment and not np.all(valid_mapping):
        raise ValueError("Corrida global cancelada: hay índices originales fuera del rango global")
    original_counts = np.bincount(original_to_global[valid_mapping], minlength=len(points))
    mapping_valid = original_counts > 0
    mapping_mismatch = np.zeros(len(original_to_global), dtype=bool)
    mapping_mismatch[valid_mapping] = (
        semantic_v1_original[valid_mapping]
        != semantic_v1[original_to_global[valid_mapping]]
    )
    mismatch_counts = np.bincount(original_to_global[mapping_mismatch], minlength=len(points))
    mapping_valid &= mismatch_counts == 0

    target_indices = choose_target_indices(points, args.roi, args.max_roi_points)
    if args.global_experiment and len(target_indices) != len(points):
        raise RuntimeError("El modo global no seleccionó la nube completa")
    if not args.global_experiment and len(target_indices) > args.max_roi_points:
        raise MemoryError("Los ROI exceden el límite de puntos objetivo")

    plan_started = time.perf_counter()
    plan = build_spatial_tile_plan(
        points,
        tile_size=args.tile_size,
        halo=args.halo,
        origin=points.min(axis=0) + np.asarray(args.origin_offset),
        max_tiles=args.max_tiles,
    )
    selected_tiles = (
        np.arange(plan.tile_count, dtype=np.int64)
        if args.global_experiment
        else tiles_touching_targets(plan, points, target_indices)
    )
    preflight = preflight_tile_plan(
        plan,
        points,
        max_tile_points=args.max_tile_points,
        max_edges=args.max_edges,
        max_total_context_points=args.max_total_context_points,
        tile_ids=selected_tiles,
    )
    memory_snapshot = process_memory_snapshot()
    estimated_tile_scratch_bytes = (
        preflight["max_context_points"] * 512
        + preflight["max_edges_upper_bound"] * 32
    )
    export_reserve_bytes = args.export_reserve_mb * 1024 * 1024
    estimated_process_peak_bytes = (
        memory_snapshot[0] + estimated_tile_scratch_bytes + export_reserve_bytes
        if memory_snapshot is not None else None
    )
    memory_budget_bytes = args.max_memory_mb * 1024 * 1024
    if args.global_experiment and memory_snapshot is None:
        raise RuntimeError("Corrida global cancelada: no se pudo medir working set para validar el límite RAM")
    if estimated_process_peak_bytes is not None and estimated_process_peak_bytes > memory_budget_bytes:
        raise MemoryError(
            f"Preflight estima {estimated_process_peak_bytes / (1024 ** 2):.1f} MiB, "
            f"sobre el límite {args.max_memory_mb} MiB"
        )
    timers["tile_plan_and_preflight_seconds"] = time.perf_counter() - plan_started

    point_count = len(points)
    target_mask = np.zeros(point_count, dtype=bool)
    target_mask[target_indices] = True
    owner_predictions = np.full(point_count, CLASS_IDS["unknown"], dtype=np.int8)
    owner_regions = np.full(point_count, -1, dtype=np.int32)
    owner_tile_ids = np.full(point_count, -1, dtype=np.int32)
    owner_counts = np.zeros(point_count, dtype=np.uint8)
    votes = np.zeros((point_count, len(CLASS_IDS)), dtype=np.uint16)
    rules = RuleConfig(decision_min_score=args.decision_min_score)
    scene_bounds = (points.min(axis=0), points.max(axis=0))
    tile_rows = []
    region_registry = []
    processing_started = time.perf_counter()

    for tile_id in selected_tiles:
        context = tile_context_indices(plan, points, int(tile_id))
        region_ids, proposals, region_metrics, region_seconds, regions = predict_context(
            context,
            points,
            normals,
            colors,
            voxel_size=args.voxel_size,
            normal_angle=args.normal_angle,
            min_region_points=args.min_region_points,
            max_tile_points=args.max_tile_points,
            max_edges=args.max_edges,
            scene_bounds=scene_bounds,
            plane_models=planes,
            vertical_axis="xyz".index(args.vertical_axis),
            vertical_confirmed=args.vertical_confirmed,
            rules=rules,
        )
        target_context = target_mask[context]
        target_rows = context[target_context]
        target_proposals = proposals[target_context]
        np.add.at(votes, (target_rows, target_proposals), 1)

        owner_in_context = plan.point_tile_ids[context] == int(tile_id)
        target_owner = target_context & owner_in_context
        owner_rows = context[target_owner]
        owner_counts[owner_rows] += 1
        owner_predictions[owner_rows] = proposals[target_owner]
        owner_regions[owner_rows] = region_ids[target_owner]
        owner_tile_ids[owner_rows] = int(tile_id)
        tile_namespace = _tile_namespace({
            "tile_size": args.tile_size,
            "halo": args.halo,
            "voxel_size": args.voxel_size,
            "normal_angle": args.normal_angle,
            "input": str(args.input.resolve()),
            "origin_xyz": plan.origin.tolist(),
            "reconciliation": args.reconciliation,
        }, plan.tile_keys[int(tile_id)])
        owner_region_predictions = proposals[target_owner]
        owner_region_ids_for_counts = region_ids[target_owner]
        per_class_points = {
            name: int(np.sum(owner_region_predictions == class_id))
            for name, class_id in CLASS_IDS.items()
        }
        region_registry.append({
            "tile_id": int(tile_id),
            "tile_key": plan.tile_keys[int(tile_id)].tolist(),
            "namespace": tile_namespace,
            "region_count_in_context": int(len(regions)),
            "owner_region_count": int(len(np.unique(owner_region_ids_for_counts[owner_region_ids_for_counts >= 0]))),
            "owner_points_by_class": per_class_points,
            "region_evidence": regions,
        })
        tile_rows.append({
            "tile_id": int(tile_id),
            "tile_key": plan.tile_keys[int(tile_id)].tolist(),
            "context_points": int(len(context)),
            "target_owner_points": int(len(owner_rows)),
            "target_context_points": int(len(target_rows)),
            "regions": int(region_metrics["regions"]),
            "context_noise_points": int(region_metrics["noise_points"]),
            "region_extraction_seconds": float(region_seconds),
            "candidate_edges": int(region_metrics["candidate_edges"]),
            "accepted_edges": int(region_metrics["accepted_edges"]),
            "rejected_edges": int(region_metrics["rejected_edges"]),
            "owner_local_proposal_counts": {
                name: int(np.sum(proposals[target_owner] == class_id))
                for name, class_id in CLASS_IDS.items()
            },
            "owner_region_size_distribution": region_size_summary(
                np.full(len(owner_rows), int(tile_id), dtype=np.int32), region_ids[target_owner],
            ),
        })
    timers["tile_inference_seconds"] = time.perf_counter() - processing_started

    final_started = time.perf_counter()
    roi_v1 = semantic_v1[target_indices]
    roi_v2, roi_sources, reconciliation_metrics = finalize_tiled_predictions(
        roi_v1,
        owner_predictions[target_indices],
        owner_regions[target_indices],
        owner_counts[target_indices],
        votes[target_indices],
        mapping_valid=mapping_valid[target_indices],
        mode=args.reconciliation,
        min_known_votes=args.min_known_votes,
    )
    projected = project_roi_to_original(
        roi_global_indices=target_indices,
        hybrid_roi_labels=roi_v2,
        hybrid_roi_sources=roi_sources,
        v1_global_labels=semantic_v1,
        v1_original_labels=semantic_v1_original,
        original_to_global=original_to_global,
    )
    if args.reconciliation == "consensus":
        known_original = projected["original_v1_labels"] != CLASS_IDS["unknown"]
        projected["original_hybrid_labels"][known_original] = projected["original_v1_labels"][known_original]
        projected["original_sources"][known_original] = SOURCE_IDS["v1"]
    timers["owner_reconciliation_and_mapping_seconds"] = time.perf_counter() - final_started

    projected_labels = projected["original_hybrid_labels"]
    projected_v1 = projected["original_v1_labels"]
    projected_sources = projected["original_sources"]
    target_comparison = compare_tile_proposals(
        owner_predictions[target_indices], votes[target_indices], owner_counts[target_indices],
    )
    target_disagreements = target_comparison["disagreement_mask"]
    target_rejection_flags = proposal_rejection_flags(
        roi_v1,
        owner_predictions[target_indices],
        owner_regions[target_indices],
        owner_counts[target_indices],
        mapping_valid[target_indices],
        target_disagreements & (roi_v2 == CLASS_IDS["unknown"]),
    )
    if args.reconciliation == "consensus":
        abstained = (roi_v1 == CLASS_IDS["unknown"]) & (roi_v2 == CLASS_IDS["unknown"])
        target_rejection_flags[abstained] |= REJECTION_REASON_BITS["consensus_abstention"]
    original_disagreements = target_disagreements[projected["local_global_rows"]]
    original_rejection_flags = target_rejection_flags[projected["local_global_rows"]]
    roi_summaries = []
    if args.roi:
        for roi_number, bounds_values in enumerate(args.roi, start=1):
            roi_global = select_roi_indices(
                points,
                (np.asarray(bounds_values[::2]), np.asarray(bounds_values[1::2])),
                max_points=args.max_roi_points,
            )
            roi_rows = np.searchsorted(target_indices, roi_global)
            roi_original_mask = np.isin(projected["local_global_rows"], roi_rows)
            roi_v1_original = projected_v1[roi_original_mask]
            roi_v2_original = projected_labels[roi_original_mask]
            roi_comparison = compare_tile_proposals(
                owner_predictions[roi_global], votes[roi_global], owner_counts[roi_global],
            )
            roi_rejection_flags = target_rejection_flags[np.searchsorted(target_indices, roi_global)]
            roi_summaries.append({
                "roi_id": roi_number,
                "bounds_xyz_half_open": [float(value) for value in bounds_values],
                "global_points": int(len(roi_global)),
                "original_points": int(np.sum(roi_original_mask)),
                "v1_geometry_coverage_percent_global": 100.0 * float(np.mean(geometry_labels[roi_global] >= 0)),
                "local_region_coverage_percent_global": 100.0 * float(np.mean(owner_regions[roi_global] >= 0)),
                "v1_counts_original": class_counts(roi_v1_original),
                "v2_counts_original": class_counts(roi_v2_original),
                "v1_coverage_percent_original": 100.0 * float(np.mean(roi_v1_original != CLASS_IDS["unknown"])) if len(roi_v1_original) else 0.0,
                "v2_coverage_percent_original": 100.0 * float(np.mean(roi_v2_original != CLASS_IDS["unknown"])) if len(roi_v2_original) else 0.0,
                "changed_original_points": int(np.sum(roi_v1_original != roi_v2_original)),
                **roi_comparison["counts"],
                "proposal_rejection_reason_counts": rejection_flag_counts(roi_rejection_flags),
                "owner_region_size_distribution": region_size_summary(
                    owner_tile_ids[roi_global], owner_regions[roi_global],
                ),
                "new_proposals_by_class_original": {
                    name: int(np.sum((projected_sources[roi_original_mask] == SOURCE_IDS["local_region"])
                                     & (roi_v2_original == class_id))
                             )
                    for name, class_id in CLASS_IDS.items()
                },
            })
    mismatches = roi_v1 != roi_v2
    local_region_sizes = region_size_summary(owner_tile_ids[target_indices], owner_regions[target_indices])
    base_config = {
        "reconciliation": args.reconciliation,
        "min_known_votes": args.min_known_votes,
        "origin_offset": list(args.origin_offset),
        "tile_size": args.tile_size,
        "halo": args.halo,
        "voxel_size": args.voxel_size,
        "normal_angle_degrees": args.normal_angle,
        "min_region_points": args.min_region_points,
        "decision_min_score": args.decision_min_score,
        "input": str(args.input.resolve()),
    }

    output_dir.mkdir(parents=True)
    export_started = time.perf_counter()
    out_cloud = make_semantic_cloud(points[target_indices], normals[target_indices], roi_v2)
    ply_path = output_dir / ("global_semantic_v2.ply" if args.global_experiment else "hybrid_tiles_roi.ply")
    if not o3d.io.write_point_cloud(str(ply_path), out_cloud):
        raise OSError(f"No se pudo exportar {ply_path}")
    npz_path = output_dir / "hybrid_tiles_labels.npz"
    target_owner_ids = owner_tile_ids[target_indices]
    target_owner_keys = np.full((len(target_indices), 3), -1, dtype=np.int32)
    valid_target_owners = target_owner_ids >= 0
    target_owner_keys[valid_target_owners] = plan.tile_keys[target_owner_ids[valid_target_owners]]
    original_global_indices = target_indices[projected["local_global_rows"]]
    original_owner_ids = owner_tile_ids[original_global_indices]
    original_owner_keys = np.full((len(original_owner_ids), 3), -1, dtype=np.int32)
    valid_original_owners = original_owner_ids >= 0
    original_owner_keys[valid_original_owners] = plan.tile_keys[original_owner_ids[valid_original_owners]]
    original_local_regions = owner_regions[original_global_indices]
    np.savez_compressed(
        npz_path,
        global_indices=target_indices,
        xyz=points[target_indices],
        rgb_original=colors[target_indices],
        normals_original=normals[target_indices],
        geometry_labels_v1=geometry_labels[target_indices],
        semantic_labels_v1_global=roi_v1,
        semantic_labels_v2_global=roi_v2,
        owner_semantic_proposal_global=owner_predictions[target_indices],
        context_vote_counts_global=votes[target_indices],
        cross_tile_disagreement_global=target_disagreements,
        proposal_rejection_reason_flags_global=target_rejection_flags,
        source_global=roi_sources,
        owner_tile_ids=target_owner_ids,
        owner_tile_keys=target_owner_keys,
        owner_local_region_ids=owner_regions[target_indices],
        original_indices=projected["original_indices"],
        original_global_indices=original_global_indices,
        semantic_labels_v1_original=projected_v1,
        semantic_labels_v2_original=projected_labels,
        cross_tile_disagreement_original=original_disagreements,
        proposal_rejection_reason_flags_original=original_rejection_flags,
        source_original=projected_sources,
        original_owner_tile_ids=original_owner_ids,
        original_owner_tile_keys=original_owner_keys,
        original_local_region_ids=original_local_regions,
        class_ids=np.asarray(list(CLASS_IDS.values()), dtype=np.int8),
        class_names=np.asarray(list(CLASS_IDS)),
        source_ids=np.asarray([SOURCE_IDS[name] for name in SOURCE_IDS], dtype=np.uint8),
        source_names=np.asarray(list(SOURCE_IDS)),
    )
    figure_path = output_dir / "tile_v1_v2_comparison.png"
    draw_tile_figure(
        figure_path,
        points[target_indices],
        roi_v1,
        roi_v2,
        owner_tile_ids[target_indices],
        owner_regions[target_indices],
        target_disagreements,
        args.seed,
    )
    roi_figure_paths = draw_roi_figures(
        output_dir,
        points[target_indices],
        roi_v1,
        roi_v2,
        owner_tile_ids[target_indices],
        owner_regions[target_indices],
        target_disagreements,
        args.roi or [],
        args.tile_size,
        plan.origin,
        args.seed,
    )
    timers["export_and_figures_seconds"] = time.perf_counter() - export_started

    report = {
        "method": "Owner tile emits final label; halo contributes context and disagreement votes only.",
        "global_experiment": bool(args.global_experiment),
        "inputs": {
            "global_cloud": str(args.input.resolve()),
            "geometry_labels_v1": str(args.geometry_labels.resolve()),
            "semantic_v1": str(args.semantic_v1.resolve()),
            "mapping": str(args.mapping.resolve()),
        },
        "parameters": {
            **base_config,
            "halo": args.halo,
            "max_tile_points": args.max_tile_points,
            "max_edges": args.max_edges,
            "max_total_context_points": args.max_total_context_points,
            "max_memory_mb": args.max_memory_mb,
            "export_reserve_mb": args.export_reserve_mb,
            "max_tiles": args.max_tiles,
            "max_roi_points": args.max_roi_points,
            "max_global_points": args.max_global_points,
            "vertical_axis": args.vertical_axis,
            "vertical_axis_confirmed": args.vertical_confirmed,
            "seed": args.seed,
            "classifier_rules": rule_config_dict(rules),
        },
        "index_integrity": {
            "global_points": point_count,
            "original_points": int(len(original_to_global)),
            "mapping_out_of_range_original_rows": int(np.sum(~valid_mapping)),
            "global_points_without_original_correspondence": int(np.sum(original_counts == 0)),
            "v1_original_global_mismatch_rows": int(mapping_mismatch.sum()),
            "target_global_points": int(len(target_indices)),
            "target_original_points": int(len(projected["original_indices"])),
            "mapping_inconsistent_target_original_rows": projected["mapping_inconsistent_rows"],
            "owner_writes_exactly_once": reconciliation_metrics["ownership_missing_or_duplicate_points"] == 0,
        },
        "tile_partition": {
            **preflight,
            "tiles_in_global_partition": int(plan.tile_count),
            "tiles_processed": int(len(selected_tiles)),
            "origin_xyz": plan.origin.tolist(),
            "roi_summaries": roi_summaries,
            "owner_rule": "floor((XYZ-origin)/tile_size), half-open interior; unique owner per global row",
            "halo_rule": "expanded owner box; context predictions never write another tile's rows",
            "halo_reconciliation": "Any semantic disagreement between owner and contexts forces unknown for V1-unknown points.",
            "instance_policy": "Region ID is namespaced by tile key and local ID; no cross-tile instance merging.",
            "tile_summaries": tile_rows,
            "region_namespace_registry": region_registry,
        },
        "v1_unknown_provenance_target": {
            "without_retained_geometric_segment": int(np.sum((geometry_labels[target_indices] < 0) & (roi_v1 == CLASS_IDS["unknown"]))),
            "unknown_inside_retained_segment": int(np.sum((geometry_labels[target_indices] >= 0) & (roi_v1 == CLASS_IDS["unknown"]))),
        },
        "geometry_coverage": {
            "v1_percent": 100.0 * float(np.mean(geometry_labels[target_indices] >= 0)),
            "local_region_percent": 100.0 * float(np.mean(owner_regions[target_indices] >= 0)),
            "local_region_noise_points": int(np.sum(owner_regions[target_indices] < 0)),
        },
        "same_original_indices_v1_v2": {
            "v1_counts": class_counts(projected_v1),
            "v2_counts": class_counts(projected_labels),
            "v1_semantic_coverage_percent": 100.0 * float(np.mean(projected_v1 != CLASS_IDS["unknown"])) if len(projected_v1) else 0.0,
            "v2_semantic_coverage_percent": 100.0 * float(np.mean(projected_labels != CLASS_IDS["unknown"])) if len(projected_labels) else 0.0,
            "v2_unknown_percent": 100.0 * float(np.mean(projected_labels == CLASS_IDS["unknown"])) if len(projected_labels) else 0.0,
            "changed_original_points": int(np.sum(projected_v1 != projected_labels)),
            "new_proposals_by_class": {
                name: int(np.sum((projected_sources == SOURCE_IDS["local_region"]) & (projected_labels == class_id)))
                for name, class_id in CLASS_IDS.items()
            },
        },
        "reconciliation": reconciliation_metrics,
        "proposal_rejection_reason_bits": REJECTION_REASON_BITS,
        "proposal_rejection_reason_counts_global": rejection_flag_counts(target_rejection_flags),
        "proposal_rejection_reason_counts_original": rejection_flag_counts(original_rejection_flags),
        "local_region_size_distribution": {
            **local_region_sizes,
        },
        "evaluation": {
            "manual_ground_truth_available": False,
            "accuracy_reported": False,
            "iou_reported": False,
            "heuristic_scores_are_probabilities": False,
        },
        "timings_seconds": timers,
        "peak_working_set_bytes": peak_working_set_bytes(),
        "preflight_memory": {
            "current_working_set_bytes": memory_snapshot[0] if memory_snapshot else None,
            "estimated_tile_scratch_bytes": int(estimated_tile_scratch_bytes),
            "export_reserve_bytes": int(export_reserve_bytes),
            "estimated_process_peak_bytes": estimated_process_peak_bytes,
            "budget_bytes": int(memory_budget_bytes),
            "estimate_method": "current working set + 512 bytes/context point + 32 bytes/potential edge + export_reserve_mb; planning estimate, not an OS-enforced hard cap",
        },
        "memory_note": "Solo se cargó global.ply, su mapeo y etiquetas; cada contexto respeta los límites por tile. El pico puede no estar disponible según la API del SO.",
        "outputs": {
            "ply": str(ply_path),
            "labels_npz": str(npz_path),
            "figure": str(figure_path),
            "roi_figures": roi_figure_paths,
        },
    }
    metrics_path = output_dir / "hybrid_tiles_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "target_global_points": len(target_indices),
        "target_original_points": len(projected["original_indices"]),
        "tiles_processed": len(selected_tiles),
        "max_tile_context_points": preflight["max_context_points"],
        "total_context_points": preflight["total_context_points"],
        "cross_tile_disagreement_points": reconciliation_metrics["cross_tile_disagreement_points"],
        "v1_semantic_coverage_percent": report["same_original_indices_v1_v2"]["v1_semantic_coverage_percent"],
        "v2_semantic_coverage_percent": report["same_original_indices_v1_v2"]["v2_semantic_coverage_percent"],
        "new_proposals_by_class": report["same_original_indices_v1_v2"]["new_proposals_by_class"],
        "peak_working_set_bytes": report["peak_working_set_bytes"],
    }, indent=2))
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, MemoryError, KeyError, IndexError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
