"""Small utilities for a bounded V1 plus local-region semantic experiment."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .semantic_classifier import CLASS_IDS, semantic_colors


SOURCE_IDS = {"v1": 0, "local_region": 1, "unknown": 2}
REJECTION_REASON_BITS = {
    "v1_known_protected": 1,
    "cross_tile_disagreement": 2,
    "invalid_mapping": 4,
    "missing_or_duplicate_owner": 8,
    "local_region_noise": 16,
    "owner_classifier_unknown_or_insufficient_evidence": 32,
    "consensus_abstention": 64,
}


@dataclass(frozen=True)
class SpatialTilePlan:
    """Owner partition and compact spatial index for incremental halo queries."""

    origin: np.ndarray
    tile_size: float
    halo: float
    tile_keys: np.ndarray
    point_tile_ids: np.ndarray
    sorted_point_indices: np.ndarray
    group_starts: np.ndarray
    group_counts: np.ndarray
    key_to_id: dict[tuple[int, int, int], int]

    @property
    def tile_count(self) -> int:
        return len(self.tile_keys)

    def owner_indices(self, tile_id: int) -> np.ndarray:
        """Indices owned by exactly this tile, in stable input order."""
        start = int(self.group_starts[tile_id])
        count = int(self.group_counts[tile_id])
        return self.sorted_point_indices[start:start + count]


def build_spatial_tile_plan(
    points: np.ndarray,
    *,
    tile_size: float,
    halo: float,
    origin: np.ndarray | None = None,
    max_tiles: int = 2_000,
) -> SpatialTilePlan:
    """Build a deterministic half-open owner partition without duplicating rows."""
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points debe ser XYZ finito con forma (n, 3)")
    if not np.isfinite(tile_size) or tile_size <= 0:
        raise ValueError("tile_size debe ser positivo y finito")
    if not np.isfinite(halo) or halo < 0 or halo > tile_size:
        raise ValueError("halo debe estar entre 0 y tile_size")
    if max_tiles < 1:
        raise ValueError("max_tiles debe ser positivo")
    tile_origin = xyz.min(axis=0) if origin is None and len(xyz) else (
        np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    )
    if tile_origin.shape != (3,) or not np.isfinite(tile_origin).all():
        raise ValueError("origin debe contener tres coordenadas finitas")
    if not len(xyz):
        return SpatialTilePlan(
            tile_origin, tile_size, halo, np.empty((0, 3), dtype=np.int64),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {},
        )

    owner_keys = assign_tile_keys(xyz, tile_origin, tile_size)
    tile_keys, point_tile_ids, group_counts = np.unique(
        owner_keys, axis=0, return_inverse=True, return_counts=True,
    )
    if len(tile_keys) > max_tiles:
        raise MemoryError(f"La partición requiere {len(tile_keys)} tiles; límite max_tiles={max_tiles}")
    point_tile_ids = point_tile_ids.astype(np.int32, copy=False)
    sorted_point_indices = np.argsort(point_tile_ids, kind="stable")
    group_starts = np.r_[0, np.cumsum(group_counts[:-1])].astype(np.int64, copy=False)
    key_to_id = {tuple(int(v) for v in key): index for index, key in enumerate(tile_keys)}
    return SpatialTilePlan(
        tile_origin, float(tile_size), float(halo), tile_keys, point_tile_ids,
        sorted_point_indices, group_starts, group_counts, key_to_id,
    )


def tile_context_indices(
    plan: SpatialTilePlan,
    points: np.ndarray,
    tile_id: int,
) -> np.ndarray:
    """Return this tile's owned rows plus halo rows, sorted by input index."""
    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(plan.point_tile_ids):
        raise ValueError("points no coincide con el plan de tiles")
    if not 0 <= tile_id < plan.tile_count:
        raise IndexError("tile_id fuera del plan")

    key = plan.tile_keys[tile_id]
    neighbor_ids = []
    # A halo no mayor que un tile can touch only the 26 adjacent owner tiles.
    for offset in product((-1, 0, 1), repeat=3):
        neighbor = tuple(int(key[axis] + offset[axis]) for axis in range(3))
        found_id = plan.key_to_id.get(neighbor)
        if found_id is not None:
            neighbor_ids.append(found_id)
    candidate_parts = [plan.owner_indices(index) for index in neighbor_ids]
    candidates = np.concatenate(candidate_parts) if candidate_parts else np.empty(0, dtype=np.int64)
    lower = plan.origin + key * plan.tile_size
    upper = lower + plan.tile_size
    expanded = np.all(
        (xyz[candidates] >= lower - plan.halo)
        & (xyz[candidates] < upper + plan.halo),
        axis=1,
    )
    return np.sort(candidates[expanded], kind="stable")


def preflight_tile_plan(
    plan: SpatialTilePlan,
    points: np.ndarray,
    *,
    max_tile_points: int,
    max_edges: int,
    max_total_context_points: int,
    tile_ids: np.ndarray | None = None,
) -> dict[str, int]:
    """Check point/edge/aggregate-work budgets before processing any tile."""
    if max_tile_points < 1 or max_edges < 1 or max_total_context_points < 1:
        raise ValueError("Los presupuestos deben ser positivos")
    selected = np.arange(plan.tile_count) if tile_ids is None else np.unique(np.asarray(tile_ids, dtype=np.int64))
    if np.any((selected < 0) | (selected >= plan.tile_count)):
        raise ValueError("tile_ids contiene índices fuera del plan")
    maximum_context = 0
    total_context = 0
    for tile_id in selected:
        rows = tile_context_indices(plan, points, int(tile_id))
        context_size = len(rows)
        maximum_context = max(maximum_context, context_size)
        total_context += context_size
        if context_size > max_tile_points:
            raise MemoryError(
                f"Tile {int(tile_id)} requiere {context_size} filas; límite max_tile_points={max_tile_points}"
            )
        # At most 13 unique undirected links per occupied voxel.
        if context_size * 13 > max_edges:
            raise MemoryError(
                f"Tile {int(tile_id)} excede el presupuesto superior de aristas max_edges={max_edges}"
            )
        if total_context > max_total_context_points:
            raise MemoryError(
                "El trabajo agregado de halo excede max_total_context_points"
            )
    return {
        "tiles_preflighted": int(len(selected)),
        "max_context_points": int(maximum_context),
        "total_context_points": int(total_context),
        "max_edges_upper_bound": int(maximum_context * 13),
    }


def finalize_tiled_predictions(
    v1_labels: np.ndarray,
    owner_predictions: np.ndarray,
    owner_region_ids: np.ndarray,
    owner_counts: np.ndarray,
    tile_votes: np.ndarray,
    *,
    mapping_valid: np.ndarray | None = None,
    mode: str = "baseline",
    min_known_votes: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Resolve with owner authority; unresolved cross-tile disagreement -> unknown."""
    baseline = np.asarray(v1_labels)
    owner = np.asarray(owner_predictions)
    region_ids = np.asarray(owner_region_ids)
    ownership = np.asarray(owner_counts)
    votes = np.asarray(tile_votes)
    count = len(baseline)
    if baseline.ndim != 1 or owner.shape != baseline.shape or region_ids.shape != baseline.shape:
        raise ValueError("Las etiquetas del propietario deben alinearse con V1")
    if ownership.shape != baseline.shape or votes.shape != (count, len(CLASS_IDS)):
        raise ValueError("Los conteos de propiedad o votos tienen forma incorrecta")
    if np.any((owner < 0) | (owner >= len(CLASS_IDS))):
        raise ValueError("Predicción de propietario fuera del catálogo semántico")
    mapping_ok = np.ones(count, dtype=bool) if mapping_valid is None else np.asarray(mapping_valid, dtype=bool)
    if mapping_ok.shape != baseline.shape:
        raise ValueError("mapping_valid no coincide con V1")

    comparison = compare_tile_proposals(owner, votes, ownership)
    exactly_one_owner = ownership == 1
    if mode not in ("baseline", "consensus") or min_known_votes < 2:
        raise ValueError("Modo o minimo de votos invalido")
    conflicts = comparison["disagreement_mask"]
    if mode == "consensus":
        known_votes = votes[:, :CLASS_IDS["unknown"]]
        distinct = np.count_nonzero(known_votes, axis=1)
        owner_known = owner != CLASS_IDS["unknown"]
        support = known_votes.sum(axis=1)
        owner_support = votes[np.arange(count), owner.astype(np.int64)]
        conflicts = (distinct > 1) | ~owner_known | (support < min_known_votes) | (owner_support < min_known_votes)
    proposals = owner.astype(np.int8, copy=True)
    proposal_regions = region_ids.astype(np.int32, copy=True)
    proposals[conflicts] = CLASS_IDS["unknown"]
    proposal_regions[conflicts] = -1
    v2, sources, fusion = fuse_unknown_only(
        baseline, proposals, proposal_regions,
        mapping_valid=mapping_ok & exactly_one_owner,
    )
    if mode == "consensus":
        known_v1 = baseline != CLASS_IDS["unknown"]
        v2[known_v1] = baseline[known_v1]
        sources[known_v1] = SOURCE_IDS["v1"]
    metrics = {
        **fusion,
        **comparison["counts"],
        "points_with_exactly_one_owner": int(exactly_one_owner.sum()),
        "ownership_missing_or_duplicate_points": int((~exactly_one_owner).sum()),
        "known_v1_predictions_overwritten": 0,
        "consensus_abstentions": int(np.sum(conflicts & (baseline == CLASS_IDS["unknown"]))) if mode == "consensus" else 0,
        "consensus_min_known_votes": min_known_votes,
    }
    return v2, sources, metrics


def compare_tile_proposals(
    owner_predictions: np.ndarray,
    tile_votes: np.ndarray,
    owner_counts: np.ndarray,
) -> dict[str, Any]:
    """Separate semantic class conflicts from known/unknown boundary changes.

    Region IDs are deliberately excluded: each tile has a local ID namespace,
    so numeric IDs from different tiles do not identify the same instance.
    """
    owner = np.asarray(owner_predictions)
    votes = np.asarray(tile_votes)
    ownership = np.asarray(owner_counts)
    if owner.ndim != 1 or ownership.shape != owner.shape:
        raise ValueError("owner_predictions y owner_counts deben ser vectores alineados")
    if votes.shape != (len(owner), len(CLASS_IDS)):
        raise ValueError("tile_votes no coincide con las predicciones del propietario")
    if not np.issubdtype(owner.dtype, np.integer) or np.any((owner < 0) | (owner >= len(CLASS_IDS))):
        raise ValueError("Predicciones del propietario fuera del catálogo")
    if np.any(votes < 0):
        raise ValueError("Los votos no pueden ser negativos")

    rows = np.arange(len(owner))
    unique_owner = ownership == 1
    owner_class_votes = votes[rows, owner.astype(np.int64)]
    owner_vote_missing = unique_owner & (owner_class_votes == 0)
    votes_without_owner = votes.copy()
    subtract_owner = unique_owner & ~owner_vote_missing
    votes_without_owner[rows[subtract_owner], owner[subtract_owner].astype(np.int64)] -= 1

    unknown_id = CLASS_IDS["unknown"]
    owner_known = owner != unknown_id
    context_unknown = votes_without_owner[:, unknown_id] > 0
    context_known_count = votes_without_owner[:, :unknown_id].sum(axis=1)
    different_known_count = context_known_count - np.where(
        owner_known, votes_without_owner[rows, owner.astype(np.int64)], 0,
    )
    class_conflict = unique_owner & owner_known & (different_known_count > 0)
    known_unknown_conflict = unique_owner & (
        (owner_known & context_unknown)
        | (~owner_known & (context_known_count > 0))
    )
    disagreement = class_conflict | known_unknown_conflict
    owner_unknown_without_known_context = unique_owner & ~owner_known & (context_known_count == 0)
    context_counts = votes_without_owner.sum(axis=1)
    multiple_context_classes = np.count_nonzero(votes_without_owner, axis=1) > 1

    return {
        "disagreement_mask": disagreement,
        "counts": {
            "cross_tile_disagreement_points": int(disagreement.sum()),
            "known_class_disagreement_points": int(class_conflict.sum()),
            "known_unknown_disagreement_points": int(known_unknown_conflict.sum()),
            "owner_unknown_without_known_context_points": int(owner_unknown_without_known_context.sum()),
            "owner_vote_missing_points": int(owner_vote_missing.sum()),
            "points_with_multiple_context_classes": int(multiple_context_classes.sum()),
            "context_predictions": int(context_counts.sum()),
            "instance_id_only_differences": None,
            "instance_ids_compared_across_tiles": False,
        },
    }


def proposal_rejection_flags(
    v1_labels: np.ndarray,
    owner_predictions: np.ndarray,
    owner_region_ids: np.ndarray,
    owner_counts: np.ndarray,
    mapping_valid: np.ndarray,
    disagreement_mask: np.ndarray,
) -> np.ndarray:
    """Return bit flags explaining why a local proposal was not applied."""
    baseline = np.asarray(v1_labels)
    owner = np.asarray(owner_predictions)
    regions = np.asarray(owner_region_ids)
    ownership = np.asarray(owner_counts)
    mapping_ok = np.asarray(mapping_valid, dtype=bool)
    disagreement = np.asarray(disagreement_mask, dtype=bool)
    if baseline.ndim != 1 or any(
        array.shape != baseline.shape
        for array in (owner, regions, ownership, mapping_ok, disagreement)
    ):
        raise ValueError("Los vectores para explicar propuestas deben estar alineados")
    flags = np.zeros(len(baseline), dtype=np.uint8)
    known_v1 = baseline != CLASS_IDS["unknown"]
    exactly_one_owner = ownership == 1
    flags[known_v1 & (owner != CLASS_IDS["unknown"])] |= REJECTION_REASON_BITS["v1_known_protected"]
    flags[disagreement] |= REJECTION_REASON_BITS["cross_tile_disagreement"]
    flags[~mapping_ok] |= REJECTION_REASON_BITS["invalid_mapping"]
    flags[~exactly_one_owner] |= REJECTION_REASON_BITS["missing_or_duplicate_owner"]
    eligible_unknown = ~known_v1 & mapping_ok & exactly_one_owner & ~disagreement
    flags[eligible_unknown & (regions < 0)] |= REJECTION_REASON_BITS["local_region_noise"]
    flags[
        eligible_unknown & (regions >= 0) & (owner == CLASS_IDS["unknown"])
    ] |= REJECTION_REASON_BITS["owner_classifier_unknown_or_insufficient_evidence"]
    return flags


def select_roi_indices(
    points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    max_points: int,
) -> np.ndarray:
    """Select XYZ rows in a contiguous half-open box [minimum, maximum)."""
    xyz = np.asarray(points)
    lower, upper = (np.asarray(value, dtype=float) for value in bounds)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points debe ser XYZ finito con forma (n, 3)")
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([*lower, *upper]).all():
        raise ValueError("Los límites del ROI deben contener seis coordenadas finitas")
    if np.any(lower >= upper):
        raise ValueError("Cada límite inferior debe ser menor que el superior")
    if max_points < 1:
        raise ValueError("max_points debe ser positivo")
    selected = np.all((xyz >= lower) & (xyz < upper), axis=1)
    indices = np.flatnonzero(selected)
    if len(indices) > max_points:
        raise MemoryError(f"El ROI tiene {len(indices)} puntos; límite max_points={max_points}")
    if not len(indices):
        raise ValueError("El ROI no contiene puntos")
    return indices


def fuse_unknown_only(
    v1_labels: np.ndarray,
    local_proposals: np.ndarray,
    local_region_ids: np.ndarray,
    *,
    mapping_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Preserve V1 and apply local proposals only to valid V1-unknown rows."""
    baseline = np.asarray(v1_labels)
    proposals = np.asarray(local_proposals)
    region_ids = np.asarray(local_region_ids)
    if baseline.ndim != 1 or proposals.shape != baseline.shape or region_ids.shape != baseline.shape:
        raise ValueError("V1, propuestas y región deben ser vectores alineados")
    if not np.issubdtype(baseline.dtype, np.integer) or not np.issubdtype(proposals.dtype, np.integer):
        raise ValueError("Las etiquetas deben ser enteras")
    valid_mapping = np.ones(len(baseline), dtype=bool) if mapping_valid is None else np.asarray(mapping_valid, dtype=bool)
    if valid_mapping.shape != baseline.shape:
        raise ValueError("mapping_valid no está alineado con las etiquetas")

    v2 = baseline.astype(np.int8, copy=True)
    source = np.where(baseline == CLASS_IDS["unknown"], SOURCE_IDS["unknown"], SOURCE_IDS["v1"]).astype(np.uint8)
    in_catalog = (proposals >= 0) & (proposals < len(CLASS_IDS))
    eligible = (
        (baseline == CLASS_IDS["unknown"])
        & (region_ids >= 0)
        & (proposals != CLASS_IDS["unknown"])
        & in_catalog
        & valid_mapping
    )
    v2[eligible] = proposals[eligible].astype(np.int8)
    source[eligible] = SOURCE_IDS["local_region"]
    # Bad mappings or malformed proposals stay unknown and are reported.
    bad_mapping = ~valid_mapping
    v2[bad_mapping] = CLASS_IDS["unknown"]
    source[bad_mapping] = SOURCE_IDS["unknown"]
    malformed = (
        (baseline == CLASS_IDS["unknown"])
        & (proposals != CLASS_IDS["unknown"])
        & ~in_catalog
    )
    v2[malformed] = CLASS_IDS["unknown"]
    source[malformed] = SOURCE_IDS["unknown"]
    metrics = {
        "local_proposals_applied": int(eligible.sum()),
        "known_v1_preserved": int(np.sum((baseline != CLASS_IDS["unknown"]) & (v2 == baseline))),
        "proposals_ignored_on_known_v1": int(np.sum((baseline != CLASS_IDS["unknown"]) & in_catalog & (proposals != CLASS_IDS["unknown"]))),
        "unknown_without_local_proposal": int(np.sum((v2 == CLASS_IDS["unknown"]) & valid_mapping)),
        "invalid_mapping_rows": int(bad_mapping.sum()),
        "invalid_proposal_rows": int(malformed.sum()),
    }
    return v2, source, metrics


def project_roi_to_original(
    *,
    roi_global_indices: np.ndarray,
    hybrid_roi_labels: np.ndarray,
    hybrid_roi_sources: np.ndarray,
    v1_global_labels: np.ndarray,
    v1_original_labels: np.ndarray,
    original_to_global: np.ndarray,
) -> dict[str, Any]:
    """Project only ROI rows to original indices; inconsistent rows become unknown."""
    roi_ids = np.asarray(roi_global_indices)
    new_labels = np.asarray(hybrid_roi_labels)
    new_sources = np.asarray(hybrid_roi_sources)
    global_v1 = np.asarray(v1_global_labels)
    original_v1 = np.asarray(v1_original_labels)
    mapping = np.asarray(original_to_global)
    if roi_ids.ndim != 1 or not np.issubdtype(roi_ids.dtype, np.integer):
        raise ValueError("roi_global_indices debe ser un vector de índices enteros")
    if new_labels.shape != roi_ids.shape or new_sources.shape != roi_ids.shape:
        raise ValueError("Las etiquetas híbridas deben alinearse con los índices del ROI")
    if (
        mapping.ndim != 1
        or mapping.shape != original_v1.shape
        or not np.issubdtype(mapping.dtype, np.integer)
    ):
        raise ValueError("El mapeo no coincide con las etiquetas originales V1")
    if np.any(roi_ids < 0) or np.any(roi_ids >= len(global_v1)):
        raise ValueError("Índice ROI fuera del espacio global")
    if len(np.unique(roi_ids)) != len(roi_ids):
        raise ValueError("Índices globales duplicados en el ROI")

    global_to_roi = np.full(len(global_v1), -1, dtype=np.int32)
    global_to_roi[roi_ids] = np.arange(len(roi_ids), dtype=np.int32)
    in_range = (mapping >= 0) & (mapping < len(global_v1))
    in_roi = np.zeros(len(mapping), dtype=bool)
    in_roi[in_range] = global_to_roi[mapping[in_range]] >= 0
    original_indices = np.flatnonzero(in_roi)
    local_rows = global_to_roi[mapping[original_indices]]

    expected_v1 = global_v1[mapping[original_indices]]
    consistent = original_v1[original_indices] == expected_v1
    labels = new_labels[local_rows].astype(np.int8, copy=True)
    sources = new_sources[local_rows].astype(np.uint8, copy=True)
    labels[~consistent] = CLASS_IDS["unknown"]
    sources[~consistent] = SOURCE_IDS["unknown"]
    return {
        "original_indices": original_indices,
        "original_v1_labels": original_v1[original_indices].copy(),
        "original_hybrid_labels": labels,
        "original_sources": sources,
        "local_global_rows": local_rows,
        "mapping_inconsistent_rows": int(np.sum(~consistent)),
        "mapping_out_of_range_rows": int(np.sum(~in_range)),
    }


def assign_tile_keys(points: np.ndarray, origin: np.ndarray, tile_size: float) -> np.ndarray:
    """Return unique half-open tile ownership keys; no tile reconciliation is run."""
    xyz = np.asarray(points, dtype=float)
    lower = np.asarray(origin, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points debe ser XYZ finito")
    if lower.shape != (3,) or not np.isfinite(lower).all() or tile_size <= 0:
        raise ValueError("origin y tile_size inválidos")
    return np.floor((xyz - lower) / tile_size).astype(np.int64)


def make_semantic_cloud(points: np.ndarray, normals: np.ndarray, labels: np.ndarray):
    """Create an Open3D ROI cloud with unchanged XYZ/normals and semantic RGB."""
    import open3d as o3d

    xyz = np.asarray(points, dtype=float)
    normal_values = np.asarray(normals, dtype=float)
    semantic = np.asarray(labels)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normal_values.shape != xyz.shape or semantic.shape != (len(xyz),):
        raise ValueError("XYZ y normales deben tener forma (n, 3)")
    if not np.isfinite(xyz).all() or not np.isfinite(normal_values).all():
        raise ValueError("XYZ y normales deben ser finitos para exportar PLY")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.normals = o3d.utility.Vector3dVector(normal_values)
    cloud.colors = o3d.utility.Vector3dVector(semantic_colors(semantic))
    return cloud
