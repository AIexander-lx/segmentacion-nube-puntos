"""Small deterministic voxel regions connected by spatial and normal affinity."""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


_HALF_NEIGHBORS = np.asarray(
    [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) > (0, 0, 0)
    ],
    dtype=np.int64,
)
_KEY_DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])


def _voxel_normal_axes(
    normals: np.ndarray,
    inverse: np.ndarray,
    voxel_count: int,
    min_coherence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate sign-invariant normal axes using summed outer products."""
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-12)
    unit = np.zeros_like(normals, dtype=float)
    unit[valid] = normals[valid] / lengths[valid, None]
    valid_counts = np.bincount(inverse[valid], minlength=voxel_count)

    # n*n.T is unchanged when a normal's sign is flipped.
    moments = np.zeros((voxel_count, 3, 3), dtype=float)
    for row in range(3):
        for col in range(row, 3):
            sums = np.bincount(
                inverse[valid],
                weights=unit[valid, row] * unit[valid, col],
                minlength=voxel_count,
            )
            moments[:, row, col] = sums
            moments[:, col, row] = sums

    eigenvalues, eigenvectors = np.linalg.eigh(moments)
    axes = eigenvectors[:, :, -1]
    coherence = eigenvalues[:, -1] / np.maximum(valid_counts, 1)
    reliable = (valid_counts > 0) & (coherence >= min_coherence)
    return axes, coherence, reliable


def _neighbor_pairs(voxel_keys: np.ndarray, max_edges: int) -> tuple[np.ndarray, np.ndarray]:
    """Find occupied 26-neighbor voxel pairs with bounded vectorized lookup."""
    voxel_count = len(voxel_keys)
    edge_upper_bound = voxel_count * len(_HALF_NEIGHBORS)
    if edge_upper_bound > max_edges:
        raise MemoryError(
            f"Presupuesto de aristas potenciales {edge_upper_bound} supera max_edges={max_edges}"
        )

    structured_keys = np.ascontiguousarray(voxel_keys).view(_KEY_DTYPE).reshape(-1)
    targets = (voxel_keys[:, None, :] + _HALF_NEIGHBORS[None, :, :]).reshape(-1, 3)
    structured_targets = np.ascontiguousarray(targets).view(_KEY_DTYPE).reshape(-1)
    positions = np.searchsorted(structured_keys, structured_targets)
    found = positions < voxel_count
    found_rows = np.flatnonzero(found)
    found[found_rows] = structured_keys[positions[found_rows]] == structured_targets[found_rows]
    source = np.repeat(np.arange(voxel_count), len(_HALF_NEIGHBORS))[found]
    return source, positions[found]


def local_voxel_regions(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    voxel_size: float = 0.20,
    normal_angle_degrees: float = 35.0,
    max_centroid_distance: float | None = None,
    min_region_points: int = 1,
    min_normal_coherence: float = 0.45,
    allow_invalid_normals: bool = True,
    max_points: int = 20_000,
    max_edges: int = 500_000,
) -> tuple[np.ndarray, dict[str, int | float | str], float]:
    """Label local voxel regions, returning one region ID per input row.

    The method first groups points by a fixed spatial grid. Occupied voxels are
    connected only to their 26 immediate neighbors when centroid distance and
    sign-invariant normal compatibility pass. Labels refer to input row order;
    ``-1`` means a voxel component smaller than ``min_region_points``.
    """
    xyz = np.asarray(points, dtype=float)
    normal_values = np.asarray(normals, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points debe ser XYZ finito con forma (n, 3)")
    if normal_values.shape != xyz.shape:
        raise ValueError("normals debe tener forma (n, 3)")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size debe ser positivo y finito")
    if not 0 <= normal_angle_degrees <= 90:
        raise ValueError("normal_angle_degrees debe estar entre 0 y 90")
    if min_region_points < 1 or max_points < 1 or max_edges < 1:
        raise ValueError("Los límites de puntos, regiones y aristas deben ser positivos")
    if not 0 <= min_normal_coherence <= 1:
        raise ValueError("min_normal_coherence debe estar entre 0 y 1")
    if max_centroid_distance is not None and max_centroid_distance <= 0:
        raise ValueError("max_centroid_distance debe ser positivo")
    if len(xyz) > max_points:
        raise MemoryError(f"El recorte tiene {len(xyz)} puntos; límite max_points={max_points}")

    started = time.perf_counter()
    count = len(xyz)
    if count == 0:
        return np.empty(0, dtype=np.int32), {
            "point_count": 0, "voxel_count": 0, "regions": 0,
            "accepted_edges": 0, "candidate_edges": 0, "noise_points": 0,
            "normal_rejected_edges": 0,
        }, 0.0

    voxel_keys, point_to_voxel, voxel_counts = np.unique(
        np.floor(xyz / voxel_size).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    voxel_count = len(voxel_keys)
    voxel_centroids = np.column_stack(
        [
            np.bincount(point_to_voxel, weights=xyz[:, axis], minlength=voxel_count)
            / voxel_counts
            for axis in range(3)
        ]
    )
    voxel_normals, normal_coherence, normal_reliable = _voxel_normal_axes(
        normal_values, point_to_voxel, voxel_count, min_normal_coherence,
    )
    source, target = _neighbor_pairs(voxel_keys, max_edges)
    candidate_count = len(source)
    centroid_distances = np.linalg.norm(voxel_centroids[source] - voxel_centroids[target], axis=1)
    distance_limit = max_centroid_distance or (2.0 * voxel_size)
    spatially_close = centroid_distances <= distance_limit
    normal_dot = np.abs(np.einsum("ij,ij->i", voxel_normals[source], voxel_normals[target]))
    normal_compatible = normal_dot >= np.cos(np.deg2rad(normal_angle_degrees))
    both_reliable = normal_reliable[source] & normal_reliable[target]
    normal_ok = np.where(both_reliable, normal_compatible, allow_invalid_normals)
    accepted = spatially_close & normal_ok

    graph = coo_matrix(
        (np.ones(int(accepted.sum()), dtype=np.uint8), (source[accepted], target[accepted])),
        shape=(voxel_count, voxel_count),
    ).tocsr()
    _, voxel_components = connected_components(graph, directed=False)
    component_sizes = np.bincount(voxel_components, weights=voxel_counts)
    keep_component = component_sizes[voxel_components] >= min_region_points
    kept_components = np.unique(voxel_components[keep_component])
    # Components are ordered by their first point, so IDs are reproducible.
    first_rows = np.full(voxel_count, count, dtype=np.int64)
    np.minimum.at(first_rows, point_to_voxel, np.arange(count))
    component_first = np.full(len(component_sizes), count, dtype=np.int64)
    np.minimum.at(component_first, voxel_components, first_rows)
    ordered_components = kept_components[np.argsort(component_first[kept_components], kind="stable")]
    component_to_region = np.full(len(component_sizes), -1, dtype=np.int32)
    component_to_region[ordered_components] = np.arange(len(ordered_components), dtype=np.int32)
    labels = component_to_region[voxel_components[point_to_voxel]]

    elapsed = time.perf_counter() - started
    region_sizes = np.bincount(labels[labels >= 0], minlength=len(ordered_components))
    metrics: dict[str, int | float | str] = {
        "method": "occupied_voxel_components_with_normal_affinity",
        "point_count": count,
        "voxel_count": voxel_count,
        "regions": len(ordered_components),
        "accepted_edges": int(accepted.sum()),
        "candidate_edges": candidate_count,
        "rejected_edges": int(candidate_count - accepted.sum()),
        "normal_rejected_edges": int((spatially_close & ~normal_ok).sum()),
        "spatially_rejected_edges": int((~spatially_close).sum()),
        "noise_points": int(np.sum(labels < 0)),
        "region_size_min": int(region_sizes.min()) if len(region_sizes) else 0,
        "region_size_median": float(np.median(region_sizes)) if len(region_sizes) else 0.0,
        "region_size_max": int(region_sizes.max()) if len(region_sizes) else 0,
        "normal_coherence_mean": float(np.mean(normal_coherence[normal_reliable])) if np.any(normal_reliable) else 0.0,
        "normal_coverage_percent": 100.0 * float(np.mean(normal_reliable)),
    }
    return labels, metrics, elapsed
