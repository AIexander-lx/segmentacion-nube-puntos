"""Interpretable geometric rules; heuristic scores are not probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CLASS_IDS = {
    "floor": 0,
    "ceiling": 1,
    "wall": 2,
    "column": 3,
    "beam": 4,
    "pipe": 5,
    "other": 6,
    "unknown": 7,
}
CLASS_COLORS = {
    "floor": [0.20, 0.70, 0.25],
    "ceiling": [0.95, 0.85, 0.20],
    "wall": [0.20, 0.45, 0.90],
    "column": [0.90, 0.30, 0.20],
    "beam": [0.75, 0.25, 0.80],
    "pipe": [0.10, 0.80, 0.85],
    "other": [0.95, 0.50, 0.10],
    "unknown": [0.45, 0.45, 0.45],
}


@dataclass(frozen=True)
class RuleConfig:
    """Decision thresholds, exposed so experiments can record them."""

    vertical_alignment_min: float = 0.80
    plane_shape_min: float = 0.55
    linearity_min: float = 0.72
    cross_section_ratio_max: float = 2.2
    normal_radial_min: float = 0.65
    pipe_radial_alignment_min: float = 0.65
    pipe_angular_coverage_min: float = 0.50
    pipe_cross_section_ratio_max: float = 2.0
    require_tube_radial_evidence: bool = False
    decision_min_score: float = 0.62
    decision_margin: float = 0.12

    def __post_init__(self) -> None:
        bounded = (
            self.vertical_alignment_min,
            self.plane_shape_min,
            self.linearity_min,
            self.normal_radial_min,
            self.pipe_radial_alignment_min,
            self.pipe_angular_coverage_min,
            self.decision_min_score,
            self.decision_margin,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("Los umbrales normalizados deben estar entre 0 y 1")
        if self.cross_section_ratio_max <= 0.0 or self.pipe_cross_section_ratio_max <= 0.0:
            raise ValueError("Los límites de proporción deben ser positivos")


def classify_region(
    features: dict[str, Any],
    *,
    region_type: str = "cluster",
    plane_model: dict[str, Any] | None = None,
    vertical_axis_confirmed: bool = False,
    config: RuleConfig = RuleConfig(),
) -> dict[str, Any]:
    """Propose one class only when geometric evidence clears simple rules."""
    class_names = [name for name in CLASS_IDS if name != "unknown"]
    scores = {name: 0.0 for name in class_names}
    evidence: dict[str, list[str]] = {name: [] for name in class_names}
    warnings: list[str] = []

    if not features.get("valid"):
        reason = features.get("reason", "invalid_features")
        return {
            "class": "unknown", "class_id": CLASS_IDS["unknown"], "score": 0.0,
            "scores": scores, "evidence": [], "warnings": [reason],
            "is_heuristic": True, "scores_are_probabilities": False,
        }
    if not vertical_axis_confirmed:
        warnings.append("vertical_axis_is_hypothesis")

    planarity = features.get("planarity") or 0.0
    linearity = features.get("linearity") or 0.0
    vertical_alignment = features.get("principal_axis_vertical_alignment") or 0.0
    vertical_ratio = features.get("vertical_to_horizontal_ratio") or 0.0
    cross_ratio = features.get("cross_section_extent_ratio")
    radial_fraction = features.get("radial_normal_fraction")
    radial_alignment = features.get("radial_normal_alignment")
    tube_cross_ratio = features.get("tube_cross_section_ratio")
    tube_angular_coverage = features.get("tube_angular_coverage")
    scene_height_fraction = features.get("scene_vertical_fraction")

    if region_type == "plane" and plane_model is not None:
        normal = plane_model.get("normal", [0.0, 0.0, 0.0])
        axis = int(features["vertical_axis_index"])
        normal_alignment = abs(float(normal[axis]))
        shape_score = min(1.0, planarity / max(config.plane_shape_min, 1e-9))

        if normal_alignment >= config.vertical_alignment_min and shape_score >= 0.65:
            at_bottom = scene_height_fraction is not None and scene_height_fraction <= 0.22
            at_top = scene_height_fraction is not None and scene_height_fraction >= 0.78
            if at_bottom:
                scores["floor"] = 0.58 + 0.28 * shape_score + 0.10 * normal_alignment
                evidence["floor"] = ["plane_normal_near_vertical_axis", "low_relative_height", "planar_region"]
            if at_top:
                scores["ceiling"] = 0.58 + 0.28 * shape_score + 0.10 * normal_alignment
                evidence["ceiling"] = ["plane_normal_near_vertical_axis", "high_relative_height", "planar_region"]
        elif normal_alignment < 0.40 and shape_score >= 0.75:
            scores["wall"] = 0.62 + 0.25 * shape_score
            evidence["wall"] = ["plane_normal_transverse_to_vertical_axis", "high_planarity"]

        # Since the vertical axis is still a hypothesis, reduce confidence.
        if not vertical_axis_confirmed:
            scores["floor"] *= 0.82
            scores["ceiling"] *= 0.82
            scores["wall"] *= 0.90
        warnings.append("plane_identity_also_depends_on_position_and_scene_orientation")
    else:
        if (
            vertical_alignment >= config.vertical_alignment_min
            and vertical_ratio >= 1.45
            and cross_ratio is not None
            and cross_ratio <= config.cross_section_ratio_max
        ):
            scores["column"] = min(
                0.88,
                0.55 + 0.18 * vertical_alignment + 0.10 * min(vertical_ratio / 3, 1) + 0.08 * planarity,
            )
            evidence["column"] = ["principal_axis_vertical", "height_dominates_horizontal_extent", "limited_cross_section"]

        if linearity >= config.linearity_min and vertical_alignment < 0.42:
            scores["beam"] = min(
                0.84,
                0.48 + 0.24 * linearity + 0.10 * (1 - vertical_alignment) + 0.08 * planarity,
            )
            evidence["beam"] = ["elongated_pca_shape", "principal_axis_transverse_to_vertical_axis"]

        # A pipe needs elongated shape, a roughly round cross-section and
        # normals that point across the principal axis. Linearity alone is weak.
        tube_shape = (
            linearity >= config.linearity_min
            and cross_ratio is not None
            and 0.55 <= cross_ratio <= 1.8
            and radial_fraction is not None
            and radial_fraction >= config.normal_radial_min
            and (
                not config.require_tube_radial_evidence
                or (
                    radial_alignment is not None
                    and radial_alignment >= config.pipe_radial_alignment_min
                    and tube_cross_ratio is not None
                    and tube_cross_ratio <= config.pipe_cross_section_ratio_max
                    and tube_angular_coverage is not None
                    and tube_angular_coverage >= config.pipe_angular_coverage_min
                )
            )
        )
        if tube_shape:
            scores["pipe"] = min(
                0.86,
                0.50 + 0.18 * linearity + 0.14 * radial_fraction + 0.08 * (1 - vertical_alignment),
            )
            evidence["pipe"] = [
                "elongated_pca_shape", "near_circular_cross_section",
                "normals_align_with_radial_vectors", "cross_section_has_angular_support",
            ]
            # A strong tube signature favors pipe over beam for an overlapping shape.
            scores["beam"] *= 0.65

        if (
            features["point_count"] >= 20
            and max(scores.values()) < 0.62
            and features.get("scattering") is not None
            and features["scattering"] > 0.08
        ):
            scores["other"] = 0.64
            evidence["other"] = ["sufficient_region_size", "no_dominant_plane_beam_column_or_tube_signature"]
        if max(scores.values()) == 0.0:
            warnings.append("insufficient_shape_evidence")

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_class, best_score = ordered[0]
    second_score = ordered[1][1]
    confident = (
        best_score >= config.decision_min_score
        and best_score - second_score >= config.decision_margin
    )
    selected = best_class if confident else "unknown"
    if not confident:
        warnings.append("ambiguous_or_low_heuristic_score")
    return {
        "class": selected,
        "class_id": CLASS_IDS[selected],
        "score": float(best_score),
        "scores": {name: float(score) for name, score in scores.items()},
        "evidence": evidence.get(selected, []),
        "best_candidate": best_class,
        "second_best_score": float(second_score),
        "warnings": warnings,
        "is_heuristic": True,
        "scores_are_probabilities": False,
    }


def propagate_semantic_labels(
    global_labels: np.ndarray,
    original_to_voxel: np.ndarray,
    *,
    unknown_id: int = CLASS_IDS["unknown"],
) -> np.ndarray:
    """Transfer classes by the existing map; invalid original rows stay unknown."""
    labels = np.asarray(global_labels)
    mapping = np.asarray(original_to_voxel)
    if labels.ndim != 1 or mapping.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Etiquetas y mapeo deben ser vectores; etiquetas enteras")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("El mapeo debe contener índices enteros")
    valid = mapping >= 0
    if np.any(mapping[valid] >= len(labels)):
        raise ValueError("Mapeo fuera de rango")
    output = np.full(len(mapping), unknown_id, dtype=np.int8)
    output[valid] = labels[mapping[valid]]
    return output


def semantic_colors(labels: np.ndarray) -> np.ndarray:
    """Return deterministic RGB colors for the fixed semantic IDs."""
    values = np.asarray(labels)
    if values.ndim != 1 or np.any((values < 0) | (values >= len(CLASS_IDS))):
        raise ValueError("ID semántico fuera del catálogo")
    palette = np.asarray([CLASS_COLORS[name] for name in CLASS_IDS], dtype=float)
    return palette[values.astype(np.int64)]


def rule_config_dict(config: RuleConfig = RuleConfig()) -> dict[str, float | bool]:
    """Serialize the rule thresholds into JSON-compatible values."""
    return asdict(config)
