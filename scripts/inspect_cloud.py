"""CLI para generar un diagnóstico JSON de una nube PLY."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentacion.io import load_ply
from segmentacion.inspection import inspect_cloud
from segmentacion.diagnostics import geometric_diagnostics

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Archivo PLY de entrada")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/metrics/cloud_inspection.json")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Máximo de puntos para figura")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible de muestreo")
    parser.add_argument("--figure", action="store_true", help="Guarda proyección XY/XZ/YZ en outputs/figures")
    parser.add_argument("--neighbor-sample", type=int, default=30000)
    args = parser.parse_args()
    try:
        cloud = load_ply(args.input)
        report = inspect_cloud(cloud)
        report["input"] = str(args.input.expanduser().resolve())
        report["visualization_sample"] = {"requested_max": args.sample_size, "seed": args.seed}
        report["geometric_diagnostics"] = geometric_diagnostics(cloud, args.neighbor_sample, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON guardado en: {args.output.resolve()}")
        if args.figure:
            if args.sample_size < 1: raise ValueError("--sample-size debe ser positivo")
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            points = np.asarray(cloud.points)
            rng = np.random.default_rng(args.seed)
            ids = rng.choice(len(points), min(len(points), args.sample_size), replace=False)
            colors = np.asarray(cloud.colors)[ids] if cloud.has_colors() else "steelblue"
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (i, j, title) in zip(axes, ((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))):
                ax.scatter(points[ids, i], points[ids, j], c=colors, s=0.15, linewidths=0, rasterized=True)
                ax.set_xlabel("XYZ"[i]); ax.set_ylabel("XYZ"[j]); ax.set_title(title); ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()
            dest = ROOT / "outputs/figures/cloud_projections.png"
            dest.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dest, dpi=180); plt.close(fig)
            print(f"Figura guardada en: {dest}")
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
if __name__ == "__main__": raise SystemExit(main())
