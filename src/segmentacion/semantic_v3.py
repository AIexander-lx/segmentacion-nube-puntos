"""Controlled local-region refinement of unknown geometric predictions.

The method processes only a selected connected component, one spatial tile at a
time. Tile interiors own points; halo points provide context only. Semantic
classes are compared across tiles, while local instance IDs are never merged.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .io import load_ply
from .local_regions import local_voxel_regions
from .semantic_classifier import (
    CLASS_COLORS,
    CLASS_IDS,
    RuleConfig,
    classify_region,
    semantic_colors,
)
from .semantic_features import region_features


SOURCE_V1 = 0
SOURCE_LOCAL = 1
SOURCE_UNKNOWN = 2
REASON_V1_PRESERVED = 0
REASON_OUTSIDE_TARGET = 1
REASON_NO_OWNER_REGION = 2
REASON_LOW_EVIDENCE = 3
REASON_CONTEXT_CONFLICT = 4
REASON_ACCEPTED = 5


def estimate_peak_memory_bytes(
    original_points: int,
    global_points: int,
    *,
    max_tile_points: int,
    max_edges: int,
    max_context_points: int,
) -> int:
    """Conservative preflight estimate; it is not an OS-enforced RAM limit."""
    values = (original_points, global_points, max_tile_points, max_edges, max_context_points)
    if any(int(value) < 0 for value in values):
        raise ValueError("Los conteos de memoria no pueden ser negativos")
    source_clouds = (original_points + global_points) * 3 * 3 * 8
    export_copy_and_colors = original_points * 12 * 8
    mappings_and_labels = original_points * 24 + global_points * 64
    bounded_tile_graph = max_tile_points * 512 + max_edges * 64 + max_context_points * 8
    process_reserve = 320 * 1024 * 1024
    return int(source_clouds + export_copy_and_colors + mappings_and_labels + bounded_tile_graph + process_reserve)


def peak_working_set_bytes() -> int | None:
    """Return the current process peak resident memory on Windows."""
    if sys.platform != "win32":
        return None

    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    try:
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        handle = ctypes.WinDLL("kernel32.dll", use_last_error=True).GetCurrentProcess()
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        return None
    return None


def _validate_alignment(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    v1_global: np.ndarray,
    v1_original: np.ndarray,
    original_to_global: np.ndarray,
    giant_indices: np.ndarray,
) -> None:
    n = len(points)
    if points.shape != (n, 3) or normals.shape != points.shape or colors.shape != points.shape:
        raise ValueError("XYZ, RGB y normales deben tener igual forma (n, 3)")
    if len(v1_global) != n or len(giant_indices) == 0:
        raise ValueError("Etiquetas V1 o componente seleccionada vacías/desalineadas")
    if original_to_global.ndim != 1 or len(v1_original) != len(original_to_global):
        raise ValueError("El mapeo global debe alinearse con las etiquetas originales")
    if not np.issubdtype(original_to_global.dtype, np.integer):
        raise ValueError("El mapeo original_to_global debe usar índices enteros")
    valid = original_to_global >= 0
    if not np.all(valid) or np.any(original_to_global[valid] >= n):
        raise ValueError("El mapeo original_to_global contiene índices inválidos")
    if len(np.unique(original_to_global)) != n:
        raise ValueError("El mapeo no cubre exactamente todos los puntos globales")
    if not np.array_equal(v1_original, v1_global[original_to_global]):
        raise ValueError("V1 original no coincide con la propagación del mapeo global")
    if not np.issubdtype(giant_indices.dtype, np.integer):
        raise ValueError("Los índices de la componente deben ser enteros")
    if np.any((giant_indices < 0) | (giant_indices >= n)) or len(np.unique(giant_indices)) != len(giant_indices):
        raise ValueError("Índices de componente duplicados o fuera de rango")


def _region_prediction(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    vertical_axis: int,
    config: RuleConfig,
    min_region_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = region_features(
        points, normals=normals, colors=colors, scene_bounds=scene_bounds,
        vertical_axis=vertical_axis,
    )
    if len(points) < min_region_points or not features.get("valid"):
        return {"class": "unknown", "score": 0.0, "evidence": [], "warnings": ["insufficient_region_support"]}, features

    # A locally planar PCA region may be evaluated as a plane. The normal is
    # the least-variance eigenvector; this is still only a geometric proposal.
    region_type = "cluster"
    plane_model = None
    if features.get("planarity") is not None and features["planarity"] >= 0.65:
        centered = points - np.mean(points, axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered / max(len(points), 1))
        plane_normal = vectors[:, 0]
        region_type = "plane"
        plane_model = {"normal": plane_normal.tolist(), "equation": [*plane_normal.tolist(), 0.0]}

    prediction = classify_region(
        features, region_type=region_type, plane_model=plane_model,
        vertical_axis_confirmed=False, config=config,
    )
    # `other` is deliberately not accepted as a fallback label in V3.
    if prediction["class"] == "other":
        prediction = dict(prediction)
        prediction["class"] = "unknown"
        prediction["class_id"] = CLASS_IDS["unknown"]
        prediction["warnings"] = list(prediction.get("warnings", [])) + ["no_specific_semantic_evidence"]
        prediction["evidence"] = []
    return prediction, features


def reconcile_context_votes(
    owner_classes: np.ndarray,
    context_votes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve class evidence; missing votes abstain and instance IDs are absent.

    A known owner proposal is retained only when known context agrees. If the
    owner abstains, unanimous known context can propose a class. Unknown context
    is represented by a row of zeros and is not a negative vote.
    """
    owner = np.asarray(owner_classes)
    votes = np.asarray(context_votes)
    if owner.ndim != 1 or votes.ndim != 2 or len(votes) != len(owner) or votes.shape[1] != len(CLASS_IDS):
        raise ValueError("owner_classes/context_votes tienen formas incompatibles")
    if not np.issubdtype(owner.dtype, np.integer) or np.any((owner < 0) | (owner >= len(CLASS_IDS))):
        raise ValueError("Las clases del propietario deben ser IDs semánticos válidos")
    if np.any(votes < 0):
        raise ValueError("Los votos de contexto no pueden ser negativos")
    known_owner = owner != CLASS_IDS["unknown"]
    vote_count = votes.sum(axis=1)
    one_context_class = (vote_count > 0) & (votes.max(axis=1) == vote_count)
    context_class = votes.argmax(axis=1).astype(np.int8)
    conflict = (vote_count > 0) & (
        ~one_context_class | (known_owner & (context_class != owner))
    )
    accepted = (known_owner & ~conflict) | (~known_owner & one_context_class)
    result = np.where(known_owner, owner, context_class).astype(np.int8)
    result[~accepted] = CLASS_IDS["unknown"]
    return result, accepted, conflict


def refine_component(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    v1_global: np.ndarray,
    v1_original: np.ndarray,
    original_to_global: np.ndarray,
    giant_indices: np.ndarray,
    instance_v1_global: np.ndarray | None = None,
    *,
    tile_size: float = 2.0,
    halo: float = 0.20,
    voxel_size: float = 0.20,
    normal_angle_degrees: float = 35.0,
    vertical_axis: int = 2,
    min_region_points: int = 20,
    max_tile_points: int = 5_000,
    max_edges: int = 100_000,
    max_tiles: int = 5_000,
    max_context_points: int = 2_000_000,
    config: RuleConfig = RuleConfig(),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Refine V1 unknowns inside one component and return V3 labels/traces.

    The deterministic half-open tile interior gives every component point one
    owner. Other tile appearances are context votes; conflicting known classes
    cause abstention. Missing/unknown predictions are not negative votes.
    """
    xyz = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.float64)
    v1g = np.asarray(v1_global)
    v1o = np.asarray(v1_original)
    mapping = np.asarray(original_to_global)
    target = np.asarray(giant_indices, dtype=np.int64)
    _validate_alignment(xyz, nrm, rgb, v1g, v1o, mapping, target)
    geometry_instances = (
        np.full(len(xyz), -1, dtype=np.int64)
        if instance_v1_global is None else np.asarray(instance_v1_global)
    )
    if geometry_instances.shape != (len(xyz),) or not np.issubdtype(geometry_instances.dtype, np.integer):
        raise ValueError("Las instancias geométricas deben ser un vector entero alineado con global")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError("XYZ/RGB de la nube global deben ser finitos")
    if vertical_axis not in (0, 1, 2):
        raise ValueError("vertical_axis debe ser 0, 1 o 2")
    if not (np.isfinite(tile_size) and tile_size > 0 and np.isfinite(voxel_size) and voxel_size > 0):
        raise ValueError("tile_size y voxel_size deben ser positivos y finitos")
    if not (np.isfinite(halo) and 0 <= halo < tile_size):
        raise ValueError("halo debe estar entre 0 y tile_size")
    if min_region_points < 1 or max_tile_points < 1 or max_edges < 1 or max_tiles < 1 or max_context_points < 1:
        raise ValueError("Los presupuestos deben ser positivos")
    if not 0 <= normal_angle_degrees <= 90:
        raise ValueError("normal_angle_degrees debe estar entre 0 y 90")

    started = time.perf_counter()
    semantic = v1g.astype(np.int8, copy=True)
    instance = geometry_instances.astype(np.int64, copy=True)
    source = np.full(len(xyz), SOURCE_UNKNOWN, dtype=np.uint8)
    source[semantic != CLASS_IDS["unknown"]] = SOURCE_V1
    reason = np.full(len(xyz), REASON_OUTSIDE_TARGET, dtype=np.uint8)
    reason[semantic != CLASS_IDS["unknown"]] = REASON_V1_PRESERVED
    # Preserve geometric instances wherever the V1 semantic class was already known.
    # The baseline geometric labels are supplied by the caller after this stage.
    giant_xyz = xyz[target]
    tile_keys = np.floor(giant_xyz / tile_size).astype(np.int64)
    unique_keys, inverse = np.unique(tile_keys, axis=0, return_inverse=True)
    tile_bins = [np.flatnonzero(inverse == i) for i in range(len(unique_keys))]
    key_to_bin = {tuple(int(v) for v in key): i for i, key in enumerate(unique_keys)}
    if len(unique_keys) > max_tiles:
        raise MemoryError(f"Componente requiere {len(unique_keys)} tiles; límite {max_tiles}")
    largest_owner = max(map(len, tile_bins), default=0)
    if largest_owner > max_tile_points:
        raise MemoryError(f"Tile propietario máximo {largest_owner} supera max_tile_points={max_tile_points}")

    # Context arrays count valid semantic votes by class. Unknown votes are
    # absent evidence and are intentionally not added to these counts.
    context_votes = np.zeros((len(xyz), len(CLASS_IDS)), dtype=np.uint16)
    owner_class = np.full(len(xyz), CLASS_IDS["unknown"], dtype=np.int8)
    owner_score = np.zeros(len(xyz), dtype=np.float32)
    owner_instance = np.full(len(xyz), -1, dtype=np.int64)
    owner_region_seen = np.zeros(len(xyz), dtype=bool)
    region_reports: list[dict[str, Any]] = []
    total_context_rows = 0
    candidate_known = 0
    accepted_edges = 0
    rejected_edges = 0
    started_regions = time.perf_counter()
    scene_bounds = (np.percentile(xyz, 0.1, axis=0), np.percentile(xyz, 99.9, axis=0))  # robusto a puntos aislados
    radius = int(np.ceil(halo / tile_size))

    for tile_number, key in enumerate(unique_keys):
        key_tuple = tuple(int(v) for v in key)
        owner_local = tile_bins[tile_number]
        neighbor_bins = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    found = key_to_bin.get((key_tuple[0] + dx, key_tuple[1] + dy, key_tuple[2] + dz))
                    if found is not None:
                        neighbor_bins.append(tile_bins[found])
        candidate_local = np.concatenate(neighbor_bins) if neighbor_bins else owner_local
        lower = key.astype(float) * tile_size - halo
        upper = (key.astype(float) + 1.0) * tile_size + halo
        candidate_xyz = giant_xyz[candidate_local]
        in_context = np.all((candidate_xyz >= lower) & (candidate_xyz < upper), axis=1)
        context_local = np.sort(candidate_local[in_context])
        total_context_rows += len(context_local)
        if total_context_rows > max_context_points:
            raise MemoryError(
                f"Puntos de contexto acumulados superan max_context_points={max_context_points}"
            )
        if len(context_local) > max_tile_points:
            raise MemoryError(
                f"Tile {key_tuple} incluye {len(context_local)} puntos con halo; límite {max_tile_points}"
            )
        global_rows = target[context_local]
        if not len(global_rows):
            continue
        local_labels, local_metrics, _ = local_voxel_regions(
            xyz[global_rows], nrm[global_rows], voxel_size=voxel_size,
            normal_angle_degrees=normal_angle_degrees, min_region_points=1,
            max_points=max_tile_points, max_edges=max_edges,
        )
        accepted_edges += int(local_metrics.get("accepted_edges", 0))
        rejected_edges += int(local_metrics.get("rejected_edges", 0))
        for local_region_id in np.unique(local_labels[local_labels >= 0]):
            rows = np.flatnonzero(local_labels == local_region_id)
            region_global = global_rows[rows]
            pred, features = _region_prediction(
                xyz[region_global], nrm[region_global], rgb[region_global],
                scene_bounds, vertical_axis, config, min_region_points,
            )
            class_id = int(pred.get("class_id", CLASS_IDS["unknown"]))
            score = float(pred.get("score", 0.0))
            is_known = class_id != CLASS_IDS["unknown"] and score >= config.decision_min_score
            owner_mask = np.all(tile_keys[context_local[rows]] == key, axis=1)
            owner_global = region_global[owner_mask]
            context_global = region_global[~owner_mask]
            # The instance ID namespace is V3-specific and does not imply that
            # fragments across tiles are the same physical object.
            region_instance = np.int64(100_000_000 + tile_number * 10_000 + int(local_region_id))
            if len(owner_global):
                owner_region_seen[owner_global] = True
                owner_class[owner_global] = class_id if is_known else CLASS_IDS["unknown"]
                owner_score[owner_global] = score
                owner_instance[owner_global] = region_instance if is_known else -1
                if is_known:
                    candidate_known += len(owner_global)
            if is_known and len(context_global):
                np.add.at(context_votes[:, class_id], context_global, 1)
            if len(owner_global):
                region_reports.append({
                    "tile": [int(x) for x in key],
                    "local_region_id": int(local_region_id),
                    "v3_instance_id": int(region_instance) if is_known else None,
                    "owner_points": int(len(owner_global)),
                    "context_points": int(len(context_global)),
                    "class": pred.get("class", "unknown"),
                    "score_heuristic": score,
                    "evidence": pred.get("evidence", []),
                    "abstention_reasons": pred.get("warnings", []) if not is_known else [],
                    "features": {
                        key: features.get(key) for key in (
                            "centroid", "extent_xyz", "principal_axis",
                            "principal_axis_vertical_alignment", "linearity", "planarity",
                            "scattering", "curvature_approx", "radial_normal_fraction",
                            "radial_normal_alignment", "tube_cross_section_ratio",
                            "tube_angular_coverage",
                            "point_count", "normals",
                        )
                    },
                })

    region_seconds = time.perf_counter() - started_regions
    target_unknown = target[semantic[target] == CLASS_IDS["unknown"]]
    votes = context_votes[target_unknown]
    owner = owner_class[target_unknown]
    owner_known = owner != CLASS_IDS["unknown"]
    context_known_count = votes.sum(axis=1)
    final_class, accepted, context_conflict = reconcile_context_votes(owner, votes)
    accepted_rows = target_unknown[accepted]
    semantic[accepted_rows] = final_class[accepted]
    accepted_owner_rows = target_unknown[accepted & owner_known]
    instance[accepted_owner_rows] = owner_instance[accepted_owner_rows]
    source[accepted_rows] = SOURCE_LOCAL
    reason[target_unknown] = REASON_NO_OWNER_REGION
    reason[target_unknown[owner_region_seen[target_unknown]]] = REASON_LOW_EVIDENCE
    reason[target_unknown[accepted]] = REASON_ACCEPTED
    reason[target_unknown[context_conflict]] = REASON_CONTEXT_CONFLICT
    reason[semantic != CLASS_IDS["unknown"]] = np.where(
        source[semantic != CLASS_IDS["unknown"]] == SOURCE_V1,
        REASON_V1_PRESERVED, REASON_ACCEPTED,
    )

    # Propagate once by the exact original_to_global map. All original rows
    # receive one semantic ID and one provenance code.
    semantic_original = semantic[mapping].astype(np.int8, copy=True)
    instance_original = instance[mapping].astype(np.int64, copy=True)
    source_original = source[mapping].astype(np.uint8, copy=True)
    reason_original = reason[mapping].astype(np.uint8, copy=True)
    elapsed = time.perf_counter() - started
    arrays = {
        "semantic_global": semantic,
        "semantic_original": semantic_original,
        "instance_global": instance,
        "instance_original": instance_original,
        "source_global": source,
        "source_original": source_original,
        "reason_global": reason,
        "reason_original": reason_original,
        "original_to_global": mapping.astype(np.int64, copy=True),
        "giant_global_indices": target,
    }
    context_comparable = context_known_count > 0
    new_instance_ids, new_instance_sizes = np.unique(instance[accepted_rows][instance[accepted_rows] >= 0], return_counts=True)
    report = {
        "method": "V1 protected; giant component refined by deterministic spatial tiles, halo, local voxel connectivity, PCA and sign-invariant normal compatibility",
        "heuristic_scores_are_probabilities": False,
        "parameters": {
            "tile_size": tile_size, "halo": halo, "local_voxel_size": voxel_size,
            "normal_angle_degrees": normal_angle_degrees, "vertical_axis_index": vertical_axis,
            "vertical_axis_confirmed": False, "min_region_points": min_region_points,
            "max_tile_points": max_tile_points, "max_edges": max_edges,
            "max_tiles": max_tiles, "max_context_points": max_context_points,
            "rule_config": {k: v for k, v in config.__dict__.items()},
        },
        "counts": {
            "global_points": int(len(xyz)), "original_points": int(len(mapping)),
            "giant_component_global": int(len(target)), "tiles": int(len(unique_keys)),
            "context_point_appearances": int(total_context_rows),
            "regions_with_owner_points": int(len(region_reports)),
            "candidate_known_owner_points_before_reconciliation": int(candidate_known),
            "context_comparable_unknown_points": int(context_comparable.sum()),
            "context_disagreement_points": int(context_conflict.sum()),
            "accepted_new_global": int(len(accepted_rows)),
            "accepted_from_owner_global": int(np.sum(accepted & owner_known)),
            "accepted_from_context_only_global": int(np.sum(accepted & ~owner_known)),
            "accepted_from_context_only_without_owner_instance_global": int(np.sum(accepted & ~owner_known)),
            "v1_known_preserved_global": int(np.sum(v1g != CLASS_IDS["unknown"])),
            "v1_known_changed_global": int(np.sum(semantic[v1g != CLASS_IDS["unknown"]] != v1g[v1g != CLASS_IDS["unknown"]])),
            "global_unknown_v1": int(np.sum(v1g == CLASS_IDS["unknown"])),
            "global_unknown_v3": int(np.sum(semantic == CLASS_IDS["unknown"])),
        },
        "class_counts_global": {name: int(np.sum(semantic == class_id)) for name, class_id in CLASS_IDS.items()},
        "class_counts_original": {name: int(np.sum(semantic_original == class_id)) for name, class_id in CLASS_IDS.items()},
        "unknown_original_percent": float(np.mean(semantic_original == CLASS_IDS["unknown"]) * 100),
        "new_instance_size_summary": {
            "instance_count": int(len(new_instance_ids)),
            "min": int(new_instance_sizes.min()) if len(new_instance_sizes) else 0,
            "median": float(np.median(new_instance_sizes)) if len(new_instance_sizes) else 0.0,
            "max": int(new_instance_sizes.max()) if len(new_instance_sizes) else 0,
            "note": "Tile-local IDs are intentionally not merged across tile boundaries.",
        },
        "source_codes": {"v1_known": SOURCE_V1, "local_region_v3": SOURCE_LOCAL, "unknown": SOURCE_UNKNOWN},
        "reason_codes": {
            "v1_known_preserved": REASON_V1_PRESERVED,
            "outside_target_component": REASON_OUTSIDE_TARGET,
            "no_owner_region": REASON_NO_OWNER_REGION,
            "low_or_invalid_geometric_evidence": REASON_LOW_EVIDENCE,
            "conflicting_context_classes": REASON_CONTEXT_CONFLICT,
            "accepted_local_proposal": REASON_ACCEPTED,
        },
        "timings_seconds": {"region_processing": region_seconds, "total_algorithm": elapsed},
        "peak_working_set_bytes": peak_working_set_bytes(),
        "regions": region_reports,
        "limitations": [
            "A tile-local instance is not asserted to match an instance in another tile.",
            "Owner evidence is preferred; unanimous known context may fill a missing owner proposal, while conflicting classes abstain. Context-only semantic proposals retain instance_id=-1.",
            "The vertical axis remains a hypothesis; heuristic scores are not probabilities.",
            "The 0.05 detail cloud was not used because this run had no separately verified refinement ROI.",
            "No human ground truth is available; semantic accuracy and IoU are not reported.",
        ],
    }
    return arrays, report


def export_v3(
    original_path: Path,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    *,
    seed: int = 42,
) -> dict[str, str]:
    """Write a new full-resolution semantic PLY, NPZ, JSON and figure."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescribe una corrida existente: {output_dir}")
    cloud = load_ply(original_path)
    semantic_original = arrays["semantic_original"]
    if len(cloud.points) != len(semantic_original):
        raise ValueError("La salida semántica no coincide con el PLY original")
    output_dir.mkdir(parents=True)
    colored = o3d.geometry.PointCloud(cloud)
    colored.colors = o3d.utility.Vector3dVector(semantic_colors(semantic_original))
    ply_path = output_dir / "semantic_v3_original.ply"
    if not o3d.io.write_point_cloud(str(ply_path), colored, write_ascii=False, compressed=False):
        raise OSError(f"No se pudo escribir {ply_path}")
    npz_path = output_dir / "semantic_v3_labels.npz"
    np.savez_compressed(
        npz_path, **arrays,
        class_ids=np.asarray(list(CLASS_IDS.values()), dtype=np.int8),
        class_names=np.asarray(list(CLASS_IDS)),
        class_colors=np.asarray([CLASS_COLORS[name] for name in CLASS_IDS], dtype=np.float32),
        provenance_codes=np.asarray([SOURCE_V1, SOURCE_LOCAL, SOURCE_UNKNOWN], dtype=np.uint8),
        provenance_names=np.asarray(["v1_known", "local_region_v3", "unknown"]),
        reason_codes=np.asarray([
            REASON_V1_PRESERVED, REASON_OUTSIDE_TARGET, REASON_NO_OWNER_REGION,
            REASON_LOW_EVIDENCE, REASON_CONTEXT_CONFLICT, REASON_ACCEPTED,
        ], dtype=np.uint8),
        reason_names=np.asarray([
            "v1_known_preserved", "outside_target_component", "no_owner_region",
            "low_or_invalid_geometric_evidence", "conflicting_context_classes",
            "accepted_local_proposal",
        ]),
    )
    figure_path = output_dir / "v1_v3_comparison.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(cloud.points)
    original_colors = np.asarray(cloud.colors) if cloud.has_colors() else np.full_like(xyz, 0.5)
    sample = np.random.default_rng(seed).choice(len(xyz), min(160_000, len(xyz)), replace=False)
    figures, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    axes[0].scatter(xyz[sample, 0], xyz[sample, 1], c=original_colors[sample], s=0.12, linewidths=0, rasterized=True)
    axes[0].set_title("RGB original (muestra)")
    axes[1].scatter(xyz[sample, 0], xyz[sample, 1], c=semantic_colors(arrays["semantic_original"])[sample], s=0.12, linewidths=0, rasterized=True)
    axes[1].set_title("Clases V3 · desconocido en gris")
    changed = arrays["source_original"][sample] == SOURCE_LOCAL
    axes[2].scatter(xyz[sample, 0], xyz[sample, 1], c=np.where(changed[:, None], [0.0, 0.75, 0.85], [0.5, 0.5, 0.5]), s=0.12, linewidths=0, rasterized=True)
    axes[2].set_title("Región local aceptada / sin identificar")
    for ax in axes:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal", adjustable="datalim")
    figures.savefig(figure_path, dpi=180)
    plt.close(figures)
    report["outputs"] = {
        "ply": str(ply_path), "npz": str(npz_path), "figure": str(figure_path),
        "ply_bytes": int(ply_path.stat().st_size), "npz_bytes": int(npz_path.stat().st_size),
    }
    report["evaluation"] = {"ground_truth_available": False, "accuracy_reported": False, "semantic_iou_reported": False}
    metrics_path = output_dir / "semantic_v3_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["outputs"]["metrics"] = str(metrics_path)
    return report["outputs"]
