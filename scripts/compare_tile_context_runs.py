"""Compare saved tiled V2 runs on identical global and original indices."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from segmentacion.semantic_classifier import CLASS_IDS, semantic_colors


CONFIGS = (
    "A_tile1_halo020",
    "B_tile2_halo040",
    "C_tile2_halo080",
    "D_tile3_halo040",
    "E_tile3_halo080",
)
ROI_3 = (10.94883406162262, 12.44883406162262,
         2.884345054626465, 4.384345054626465,
         1.16875559091568, 2.66875559091568)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", type=float, nargs=6, default=ROI_3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_data = {}
    reference_indices = None
    reference_original_indices = None
    reference_v1 = None
    reference_v1_global = None
    reference_xyz = None
    for name in CONFIGS:
        directory = args.runs_dir / name
        metrics_path = directory / "hybrid_tiles_metrics.json"
        labels_path = directory / "hybrid_tiles_labels.npz"
        if not metrics_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Falta la salida completa para {name}: {directory}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(labels_path) as archive:
            labels = {key: archive[key].copy() for key in (
                "global_indices", "original_indices", "semantic_labels_v1_global",
                "semantic_labels_v2_global", "semantic_labels_v1_original",
                "semantic_labels_v2_original", "original_global_indices",
                "source_original", "owner_tile_ids", "original_owner_tile_ids", "xyz",
                "cross_tile_disagreement_global",
                "proposal_rejection_reason_flags_original",
            )}
        if not metrics["index_integrity"]["owner_writes_exactly_once"]:
            raise ValueError(f"{name} reporta propietarios ausentes o duplicados")
        if metrics["index_integrity"]["mapping_out_of_range_original_rows"] or metrics["index_integrity"]["v1_original_global_mismatch_rows"]:
            raise ValueError(f"{name} reporta inconsistencias entre mapeo y etiquetas V1")
        mapping_path = Path(metrics["inputs"]["mapping"])
        with np.load(mapping_path) as mapping_archive:
            original_to_global = mapping_archive["original_to_voxel"]
            mapped_global = original_to_global[labels["original_indices"]]
        if not np.array_equal(mapped_global, labels["original_global_indices"]):
            raise ValueError(f"{name} NPZ no conserva original_to_voxel")
        global_rows = np.searchsorted(labels["global_indices"], mapped_global)
        if not np.array_equal(labels["semantic_labels_v2_original"], labels["semantic_labels_v2_global"][global_rows]):
            raise ValueError(f"{name} etiquetas originales no corresponden a los globales")
        known_v1 = labels["semantic_labels_v1_global"] != CLASS_IDS["unknown"]
        if not np.array_equal(labels["semantic_labels_v1_global"][known_v1], labels["semantic_labels_v2_global"][known_v1]):
            raise ValueError(f"{name} altera una etiqueta V1 conocida")
        if int(labels["cross_tile_disagreement_global"].sum()) != metrics["reconciliation"]["cross_tile_disagreement_points"]:
            raise ValueError(f"{name} máscara de desacuerdo no coincide con sus métricas")
        if np.any(labels["owner_tile_ids"] < 0) or np.any(labels["original_owner_tile_ids"] < 0):
            raise ValueError(f"{name} tiene filas objetivo sin tile propietario")
        if len(labels["proposal_rejection_reason_flags_original"]) != len(labels["original_indices"]):
            raise ValueError(f"{name} banderas de rechazo no alineadas con los originales")
        reason_bits = metrics["proposal_rejection_reason_bits"]
        for reason, bit in reason_bits.items():
            measured = int(np.sum((labels["proposal_rejection_reason_flags_original"] & int(bit)) != 0))
            if measured != metrics["proposal_rejection_reason_counts_original"][reason]:
                raise ValueError(f"{name} conteo de razón {reason} no coincide con NPZ")
        if reference_indices is None:
            reference_indices = labels["global_indices"]
            reference_original_indices = labels["original_indices"]
            reference_v1 = labels["semantic_labels_v1_original"]
            reference_v1_global = labels["semantic_labels_v1_global"]
            reference_xyz = labels["xyz"]
        elif not (
            np.array_equal(reference_indices, labels["global_indices"])
            and np.array_equal(reference_original_indices, labels["original_indices"])
            and np.array_equal(reference_v1, labels["semantic_labels_v1_original"])
            and np.array_equal(reference_v1_global, labels["semantic_labels_v1_global"])
            and np.array_equal(reference_xyz, labels["xyz"])
        ):
            raise ValueError(f"{name} no usa exactamente los mismos índices originales/globales y V1")
        run_data[name] = {"metrics": metrics, "labels": labels}

    per_config = {}
    for name, data in run_data.items():
        metrics = data["metrics"]
        labels = data["labels"]
        per_config[name] = {
            "parameters": metrics["parameters"],
            "global_indices": int(len(labels["global_indices"])),
            "original_indices": int(len(labels["original_indices"])),
            "v1_class_counts_original": metrics["same_original_indices_v1_v2"]["v1_counts"],
            "v2_class_counts_original": metrics["same_original_indices_v1_v2"]["v2_counts"],
            "v1_coverage_percent_original": metrics["same_original_indices_v1_v2"]["v1_semantic_coverage_percent"],
            "v2_coverage_percent_original": metrics["same_original_indices_v1_v2"]["v2_semantic_coverage_percent"],
            "v2_unknown_percent_original": metrics["same_original_indices_v1_v2"]["v2_unknown_percent"],
            "changed_original_points_vs_v1": metrics["same_original_indices_v1_v2"]["changed_original_points"],
            "new_proposals_by_class_original": metrics["same_original_indices_v1_v2"]["new_proposals_by_class"],
            "tile_count_full_plan": metrics["tile_partition"]["tiles_in_global_partition"],
            "tiles_processed_for_rois": metrics["tile_partition"]["tiles_processed"],
            "max_context_points": metrics["tile_partition"]["max_context_points"],
            "total_context_points": metrics["tile_partition"]["total_context_points"],
            "owner_region_size_distribution": metrics["local_region_size_distribution"],
            "timings_seconds": metrics["timings_seconds"],
            "peak_working_set_bytes": metrics["peak_working_set_bytes"],
            "preflight_memory": metrics.get("preflight_memory"),
            "reconciliation": metrics["reconciliation"],
            "proposal_rejection_reason_bits": metrics["proposal_rejection_reason_bits"],
            "proposal_rejection_reason_counts_original": metrics["proposal_rejection_reason_counts_original"],
            "roi_summaries": metrics["tile_partition"]["roi_summaries"],
        }

    pairwise = []
    for left, right in combinations(CONFIGS, 2):
        first = run_data[left]["labels"]["semantic_labels_v2_original"]
        second = run_data[right]["labels"]["semantic_labels_v2_original"]
        first_known = first != CLASS_IDS["unknown"]
        second_known = second != CLASS_IDS["unknown"]
        known_known = first_known & second_known
        known_class_difference = known_known & (first != second)
        known_unknown = first_known ^ second_known
        pairwise.append({
            "left": left,
            "right": right,
            "same_original_indices": True,
            "semantic_difference_points": int(np.sum(first != second)),
            "known_class_difference_points": int(known_class_difference.sum()),
            "known_unknown_difference_points": int(known_unknown.sum()),
            "instance_id_only_differences": None,
            "instance_ids_compared": False,
        })

    lower = np.asarray(args.roi[::2], dtype=float)
    upper = np.asarray(args.roi[1::2], dtype=float)
    figure, axes = plt.subplots(2, len(CONFIGS), figsize=(4 * len(CONFIGS), 8), squeeze=False)
    roi_rows = None
    for column, name in enumerate(CONFIGS):
        data = run_data[name]
        labels = data["labels"]
        xyz = labels["xyz"]
        if roi_rows is None:
            roi_rows = np.all((xyz >= lower) & (xyz < upper), axis=1)
        roi_xyz = xyz[roi_rows]
        roi_semantic = labels["semantic_labels_v2_global"][roi_rows]
        roi_conflict = labels["cross_tile_disagreement_global"][roi_rows]
        roi_report = next((
            row for row in data["metrics"]["tile_partition"]["roi_summaries"]
            if np.allclose(row["bounds_xyz_half_open"], args.roi, rtol=0, atol=1e-6)
        ), None)
        disagreement_count = int(roi_conflict.sum())
        if roi_report is not None:
            disagreement_count = roi_report["cross_tile_disagreement_points"]
        coverage = float(np.mean(roi_semantic != CLASS_IDS["unknown"])) * 100 if len(roi_semantic) else 0.0
        axes[0, column].scatter(roi_xyz[:, 0], roi_xyz[:, 2], c=semantic_colors(roi_semantic), s=3, linewidths=0, rasterized=True)
        axes[0, column].set_title(f"{name}\nV2 ROI: {coverage:.1f}% cobertura")
        axes[1, column].scatter(
            roi_xyz[:, 0], roi_xyz[:, 2],
            c=np.where(roi_conflict[:, None], [0.95, 0.05, 0.05], [0.55, 0.55, 0.55]),
            s=3, linewidths=0, rasterized=True,
        )
        axes[1, column].set_title(f"Desacuerdo: {disagreement_count}/{len(roi_xyz)}")
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xlabel("X")
            axis.set_ylabel("Z")
            axis.set_aspect("equal", adjustable="box")
    figure.suptitle("ROI alargado: propuestas V2 y discrepancias owner/contexto por configuración")
    figure.tight_layout()
    if args.output_dir.exists():
        raise FileExistsError(f"No se sobrescriben resúmenes anteriores: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "roi_03_context_comparison.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    report = {
        "comparison_method": "Only semantic labels on identical original indices are compared; local instance IDs are namespaced and excluded.",
        "same_global_indices": True,
        "same_original_indices": True,
        "same_v1_original_labels": True,
        "same_v1_global_labels_and_xyz": True,
        "instance_id_only_differences": None,
        "per_configuration": per_config,
        "pairwise_configuration_differences": pairwise,
        "roi_figure": str(figure_path.resolve()),
        "ground_truth_available": False,
        "accuracy_or_iou_reported": False,
    }
    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_path.resolve()}")
    print(f"ROI figure: {figure_path.resolve()}")
    print(json.dumps({
        name: {
            "tiles": data["metrics"]["tile_partition"]["tiles_in_global_partition"],
            "max_context": data["metrics"]["tile_partition"]["max_context_points"],
            "context_points": data["metrics"]["tile_partition"]["total_context_points"],
            "coverage_v2": data["metrics"]["same_original_indices_v1_v2"]["v2_semantic_coverage_percent"],
            "disagreement": data["metrics"]["reconciliation"]["cross_tile_disagreement_points"],
            "memory_peak": data["metrics"]["peak_working_set_bytes"],
        }
        for name, data in run_data.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
