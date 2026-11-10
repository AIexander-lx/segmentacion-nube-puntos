"""Compact, interpretable geometric descriptors for semantic proposals."""
from __future__ import annotations

from typing import Any

import numpy as np


def region_features(
    points: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    vertical_axis: int = 2,
    plane_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize shape, pose, normals, RGB and distance to known planes.

    Density is points divided by the axis-aligned bounding-box volume. It is
    only a rough descriptor; it is not an estimate of the object's volume.
    """
    raw_points = np.asarray(points, dtype=float)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ValueError("points debe tener forma (n, 3)")
    if vertical_axis not in (0, 1, 2):
        raise ValueError("vertical_axis debe ser 0, 1 o 2")

    finite_point_mask = np.isfinite(raw_points).all(axis=1)
    xyz = raw_points[finite_point_mask]
    count = len(xyz)
    if count == 0:
        return {"point_count": 0, "valid": False, "reason": "empty_or_nonfinite_xyz"}

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    extent = xyz_max - xyz_min
    centroid = xyz.mean(axis=0)
    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
    horizontal_extent = float(max(extent[horizontal_axes]))
    vertical_extent = float(extent[vertical_axis])

    # Covariance eigenvalues describe the spread along the principal axes.
    centered = xyz - centroid
    covariance = centered.T @ centered / max(count, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    principal_axis = eigenvectors[:, -1]
    largest = float(eigenvalues[0]) if count >= 3 else 0.0
    total_spread = float(eigenvalues.sum())

    if largest > 1e-15 and total_spread > 1e-15:
        linearity = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
        planarity = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])
        scattering = float(eigenvalues[2] / eigenvalues[0])
        curvature = float(eigenvalues[2] / total_spread)
        vertical_alignment = float(abs(principal_axis[vertical_axis]))
    else:
        principal_axis = np.zeros(3)
        linearity = planarity = scattering = curvature = vertical_alignment = None

    box_volume = float(np.prod(extent))
    box_density = float(count / box_volume) if box_volume > 1e-12 else None
    sorted_extent = np.sort(extent)[::-1]
    cross_section_ratio = (
        float(sorted_extent[1] / sorted_extent[2])
        if sorted_extent[2] > 1e-12 else None
    )

    normal_summary: dict[str, Any] | None = None
    radial_normal_fraction: float | None = None
    radial_normal_alignment: float | None = None
    tube_cross_section_ratio: float | None = None
    tube_angular_coverage: float | None = None
    if largest > 1e-15:
        axis = principal_axis / max(float(np.linalg.norm(principal_axis)), 1e-12)
        transverse = centered - np.outer(centered @ axis, axis)
        transverse_covariance = transverse.T @ transverse / max(count, 1)
        cross_eigenvalues = np.maximum(np.linalg.eigvalsh(transverse_covariance)[::-1], 0.0)
        if cross_eigenvalues[1] > 1e-15:
            tube_cross_section_ratio = float(np.sqrt(cross_eigenvalues[0] / cross_eigenvalues[1]))
        basis_x = eigenvectors[:, 1]
        basis_y = np.cross(axis, basis_x)
        radial_distance = np.linalg.norm(transverse, axis=1)
        radial_valid = radial_distance > 1e-12
        if np.any(radial_valid):
            angle = np.arctan2(transverse[radial_valid] @ basis_y, transverse[radial_valid] @ basis_x)
            bins = np.floor((angle + np.pi) * (8.0 / (2.0 * np.pi))).astype(np.int64) % 8
            tube_angular_coverage = float(len(np.unique(bins)) / 8.0)
    if normals is not None:
        raw_normals = np.asarray(normals, dtype=float)
        if raw_normals.shape == raw_points.shape:
            raw_normals = raw_normals[finite_point_mask]
        if raw_normals.shape == xyz.shape:
            lengths = np.linalg.norm(raw_normals, axis=1)
            valid_normals = np.isfinite(raw_normals).all(axis=1) & (lengths > 1e-12)
            unit_normals = raw_normals[valid_normals] / lengths[valid_normals, None]
            normal_summary = {
                "valid_count": int(valid_normals.sum()),
                "invalid_count": int(count - valid_normals.sum()),
                "mean_unit": unit_normals.mean(axis=0).tolist() if len(unit_normals) else None,
                "axis_abs_fraction_ge_0_9": (
                    np.mean(np.abs(unit_normals) >= 0.9, axis=0).tolist()
                    if len(unit_normals) else None
                ),
            }
            if len(unit_normals) and largest > 1e-15:
                # For a tube, surface normals tend to be transverse to its axis.
                radial_normal_fraction = float(
                    np.mean(np.abs(unit_normals @ principal_axis) < 0.35)
                )
                normalized_axis = principal_axis / max(float(np.linalg.norm(principal_axis)), 1e-12)
                valid_xyz = xyz[valid_normals]
                radial = valid_xyz - centroid - np.outer((valid_xyz - centroid) @ normalized_axis, normalized_axis)
                radial_lengths = np.linalg.norm(radial, axis=1)
                has_radial = radial_lengths > 1e-12
                if np.any(has_radial):
                    radial_unit = radial[has_radial] / radial_lengths[has_radial, None]
                    radial_normal_alignment = float(
                        np.mean(np.abs(np.einsum("ij,ij->i", unit_normals[has_radial], radial_unit)) >= 0.65)
                    )
        else:
            normal_summary = {"valid_count": 0, "invalid_count": count, "reason": "shape_mismatch"}

    rgb_summary: dict[str, Any] | None = None
    if colors is not None:
        raw_colors = np.asarray(colors, dtype=float)
        if raw_colors.shape == raw_points.shape:
            raw_colors = raw_colors[finite_point_mask]
        if raw_colors.shape == xyz.shape:
            valid_colors = np.isfinite(raw_colors).all(axis=1)
            rgb = raw_colors[valid_colors]
            rgb_summary = {
                "valid_count": int(valid_colors.sum()),
                "mean": rgb.mean(axis=0).tolist() if len(rgb) else None,
                "std": rgb.std(axis=0).tolist() if len(rgb) else None,
                "min": rgb.min(axis=0).tolist() if len(rgb) else None,
                "max": rgb.max(axis=0).tolist() if len(rgb) else None,
            }

    scene_vertical_fraction = None
    if scene_bounds is not None:
        scene_min, scene_max = (np.asarray(bound, dtype=float) for bound in scene_bounds)
        scene_height = float(scene_max[vertical_axis] - scene_min[vertical_axis])
        if scene_height > 1e-12:
            scene_vertical_fraction = float(
                (centroid[vertical_axis] - scene_min[vertical_axis]) / scene_height
            )

    plane_distances = []
    for model in plane_models or []:
        equation = np.asarray(model.get("equation", []), dtype=float)
        normal_length = np.linalg.norm(equation[:3]) if equation.shape == (4,) else 0.0
        if equation.shape == (4,) and normal_length > 1e-12:
            distances = np.abs(xyz @ equation[:3] + equation[3]) / normal_length
            plane_distances.append({
                "plane_id": model.get("id"),
                "median_abs_distance": float(np.median(distances)),
            })

    return {
        "point_count": count,
        "valid": True,
        "centroid": centroid.tolist(),
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        "extent_xyz": extent.tolist(),
        "vertical_axis_index": vertical_axis,
        "vertical_extent": vertical_extent,
        "horizontal_max_extent": horizontal_extent,
        "vertical_to_horizontal_ratio": float(vertical_extent / max(horizontal_extent, 1e-12)),
        "extent_sorted_desc": sorted_extent.tolist(),
        "pca_eigenvalues": eigenvalues.tolist(),
        "principal_axis": principal_axis.tolist(),
        "principal_axis_vertical_alignment": vertical_alignment,
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "curvature_approx": curvature,
        "cross_section_extent_ratio": cross_section_ratio,
        "radial_normal_fraction": radial_normal_fraction,
        "radial_normal_alignment": radial_normal_alignment,
        "tube_cross_section_ratio": tube_cross_section_ratio,
        "tube_angular_coverage": tube_angular_coverage,
        "normals": normal_summary,
        "rgb_auxiliary": rgb_summary,
        "bounding_box_volume": box_volume,
        "points_per_bounding_box_volume": box_density,
        "density_note": "count / axis-aligned bounding-box volume; not physical object volume",
        "scene_vertical_fraction": scene_vertical_fraction,
        "plane_distances": plane_distances,
        "degenerate_geometry": largest <= 1e-15,
    }
