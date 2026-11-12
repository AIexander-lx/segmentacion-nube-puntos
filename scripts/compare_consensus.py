"""Compare three saved ROI runs using their identical point indices."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from segmentacion.hybrid_pipeline import make_semantic_cloud
from segmentacion.semantic_classifier import semantic_colors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--consensus", type=Path, required=True)
    p.add_argument("--shifted", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    runs = {}
    for name in ("baseline", "consensus", "shifted"):
        directory = getattr(a, name)
        with np.load(directory / "hybrid_tiles_labels.npz") as z:
            arrays = {k: z[k].copy() for k in z.files}
        metrics = json.loads((directory / "hybrid_tiles_metrics.json").read_text(encoding="utf-8"))
        runs[name] = (arrays, metrics)
    reference = runs["baseline"][0]
    for z, m in runs.values():
        for key in ("global_indices", "original_indices", "semantic_labels_v1_original", "xyz"):
            if not np.array_equal(z[key], reference[key]):
                raise ValueError(f"Indices no alineados: {key}")
        if not m["index_integrity"]["owner_writes_exactly_once"] or m["index_integrity"]["mapping_inconsistent_target_original_rows"]:
            raise ValueError("Propiedad o mapeo invalido")
        with np.load(m["inputs"]["mapping"]) as mapping:
            if not np.array_equal(mapping["original_to_voxel"][z["original_indices"]], z["original_global_indices"]):
                raise ValueError("La correspondencia original a global no coincide")
        known = z["semantic_labels_v1_original"] != 7
        if not np.array_equal(z["semantic_labels_v1_original"][known], z["semantic_labels_v2_original"][known]):
            raise ValueError("Se altero una clase V1 conocida")
    a.output_dir.mkdir(parents=True)
    stats = {}
    for name, (z, m) in runs.items():
        labels = z["semantic_labels_v2_original"]
        stats[name] = {"coverage_percent": float(np.mean(labels != 7)*100),
                       "new_original_points": int(np.sum((z["semantic_labels_v1_original"] == 7) & (labels != 7))),
                       "unknown_original_points": int(np.sum(labels == 7)),
                       "disagreement_global": m["reconciliation"]["cross_tile_disagreement_points"],
                       "abstentions_global": m["reconciliation"].get("consensus_abstentions", 0),
                       "timings_seconds": m["timings_seconds"], "peak_bytes": m["peak_working_set_bytes"],
                       "region_sizes": m["local_region_size_distribution"]}
        for roi in m["tile_partition"]["roi_summaries"]:
            bounds = np.asarray(roi["bounds_xyz_half_open"])
            mask = np.all((z["xyz"] >= bounds[::2]) & (z["xyz"] < bounds[1::2]), axis=1)
            cloud = make_semantic_cloud(z["xyz"][mask], z["normals_original"][mask], z["semantic_labels_v2_global"][mask])
            if not o3d.io.write_point_cloud(str(a.output_dir / f"{name}_roi{roi['roi_id']}.ply"), cloud):
                raise OSError("No se pudo guardar PLY")
    pairs = {}
    for left, right in (("baseline", "consensus"), ("consensus", "shifted"), ("baseline", "shifted")):
        x = runs[left][0]["semantic_labels_v2_original"]
        y = runs[right][0]["semantic_labels_v2_original"]
        pairs[left + "_vs_" + right] = {"different_original_points": int(np.sum(x != y)),
            "known_class_contradictions": int(np.sum((x != y) & (x != 7) & (y != 7))),
            "known_unknown_changes": int(np.sum((x != 7) ^ (y != 7)))}
    xyz = reference["xyz"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    data = [("V1", reference["semantic_labels_v1_global"])] + [(n, runs[n][0]["semantic_labels_v2_global"]) for n in runs]
    for axis, (name, labels) in zip(axes, data):
        axis.scatter(xyz[:, 0], xyz[:, 2], c=semantic_colors(labels), s=2, linewidths=0)
        axis.set_title(name); axis.set_xlabel("X"); axis.set_ylabel("Z")
    fig.tight_layout(); fig.savefig(a.output_dir / "comparison.png", dpi=170); plt.close(fig)
    # Report region descriptors for the same global points whose final labels differ.
    cases = []
    z1, m1 = runs["consensus"]; z2, m2 = runs["shifted"]
    changed = np.flatnonzero(z1["semantic_labels_v2_global"] != z2["semantic_labels_v2_global"])
    registries = []
    for metrics in (m1, m2):
        registries.append({(row["tile_id"], region["region_id_local"]): region
                           for row in metrics["tile_partition"]["region_namespace_registry"]
                           for region in row.get("region_evidence", [])})
    for row in changed[:30]:
        evidence = []
        for z, registry in ((z1, registries[0]), (z2, registries[1])):
            evidence.append(registry.get((int(z["owner_tile_ids"][row]), int(z["owner_local_region_ids"][row]))))
        cases.append({"global_index": int(reference["global_indices"][row]), "xyz": xyz[row].tolist(),
                      "consensus_label": int(z1["semantic_labels_v2_global"][row]),
                      "shifted_label": int(z2["semantic_labels_v2_global"][row]),
                      "consensus_votes": z1["context_vote_counts_global"][row].tolist(),
                      "shifted_votes": z2["context_vote_counts_global"][row].tolist(),
                      "region_evidence": evidence})
    report = {"same_indices": True, "runs": stats, "pairwise": pairs, "geometry_cases": cases,
              "note": "Scores son heuristicas; diferencias no demuestran precision."}
    (a.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"runs": stats, "pairwise": pairs, "geometry_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
