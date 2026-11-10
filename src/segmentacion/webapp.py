"""Local read-only viewer and bounded processing jobs for the factory project."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import open3d as o3d

from .io import load_ply
from .semantic_classifier import CLASS_COLORS, CLASS_IDS, semantic_colors
from .visualization import label_colors, normals_colors

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "web"
ORIGINAL_PLY_NAME = "FARO_Scan_Full_Sub1_down3_with_normals.ply"
CLOUDS = {"original": ROOT / ORIGINAL_PLY_NAME, "global": ROOT / "outputs/processed/global.ply", "detail": ROOT / "outputs/processed/detail.ply"}
GEOMETRY = ROOT / "outputs/segmented/global_review_final/labels.npz"
SEMANTIC = ROOT / "outputs/semantic/global_heuristic_v1/semantic_labels.npz"
MAX_POINTS = 500_000
MAX_TRIANGLES = 100_000


class Project:
    """One cached resolution, with stable original-index mappings for labels."""

    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.token = secrets.token_urlsafe(24)
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.cached_name = None
        self.cached_cloud = None
        self.custom_input: Path | None = None
        self.job = {"state": "idle", "message": "Sin procesamiento activo"}
        self.job_started: float | None = None
        self._result_stats_cache: dict[str, tuple[int, dict]] = {}

    def path(self, path: Path) -> Path:
        relative = path.relative_to(ROOT)
        candidate = self.root / relative
        if relative == Path(ORIGINAL_PLY_NAME) and not candidate.is_file():
            raw_copy = self.root / "data/raw" / ORIGINAL_PLY_NAME
            if raw_copy.is_file():
                return raw_copy
        return candidate

    def meshes(self) -> dict[str, Path]:
        root = self.root / "outputs/reconstruction"
        found = {"reconstruction/" + str(p.relative_to(root)).replace("\\", "/"): p for p in sorted(root.rglob("*.ply")) if p.is_file()} if root.is_dir() else {}
        jobs = self.root / "outputs/app_runs"
        if jobs.is_dir():
            found.update({"app_runs/" + str(p.relative_to(jobs)).replace("\\", "/"): p for p in sorted(jobs.rglob("*.ply")) if p.name in ("ball_pivoting.ply", "poisson.ply")})
        return found

    def metadata(self) -> dict:
        stats = {}
        for name, path in (("geometry", self.root / "outputs/segmented/global_review_final/metrics.json"), ("semantic", self.root / "outputs/semantic/global_heuristic_v1/semantic_metrics.json")):
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    stats[name] = {key: data[key] for key in ("counts", "coverage", "timings_seconds") if key in data}
                except (OSError, ValueError):
                    stats[name] = {"error": "No se pudo leer el reporte"}
        v3_metrics = self.root / "outputs/semantic/global_local_v3_release/semantic_v3_metrics.json"
        if v3_metrics.is_file():
            try:
                data = json.loads(v3_metrics.read_text(encoding="utf-8"))
                stats["semantic_v3"] = {
                    key: data[key] for key in (
                        "comparison_v1", "comparison_v3", "class_counts_original",
                        "class_counts_global", "timings_seconds",
                    ) if key in data
                }
            except (OSError, ValueError):
                stats["semantic_v3"] = {"error": "No se pudo leer el reporte V3"}
        clouds = {k: self.path(v).is_file() for k, v in CLOUDS.items()}
        if self.custom_input is not None: clouds = {"original": self.custom_input.is_file(), "global": False, "detail": False}
        job = self.job.copy()
        if job["state"] == "running" and self.job_started is not None:
            job["elapsed_seconds"] = round(time.monotonic() - self.job_started, 1)
        result_stats = {}
        for name, labels_path in self.results().items():
            metric_candidates = (
                labels_path.parent / "semantic_v3_metrics.json",
                labels_path.parent / "hybrid_tiles_metrics.json",
                labels_path.parent / "metrics.json",
            )
            metric_path = next((candidate for candidate in metric_candidates if candidate.is_file()), None)
            if metric_path is not None:
                try:
                    modified = metric_path.stat().st_mtime_ns
                    cached = self._result_stats_cache.get(str(metric_path))
                    if cached is not None and cached[0] == modified:
                        result_stats[name] = cached[1]
                    else:
                        metric_data = json.loads(metric_path.read_text(encoding="utf-8"))
                        summary = {
                            key: metric_data[key] for key in (
                                "comparison_v1", "comparison_v3", "class_counts_original",
                                "class_counts_global", "timings_seconds", "coverage", "counts",
                            ) if key in metric_data
                        }
                        self._result_stats_cache[str(metric_path)] = (modified, summary)
                        result_stats[name] = summary
                except (OSError, ValueError):
                    result_stats[name] = {"error": "No se pudo leer el reporte de esta corrida"}
        return {"clouds": clouds, "custom_input": self.custom_input.name if self.custom_input else None,
                "instances_available": self.custom_input is None and self.path(GEOMETRY).is_file(), "semantic_available": self.custom_input is None and self.path(SEMANTIC).is_file(),
                "meshes": list(self.meshes()), "classes": CLASS_COLORS, "token": self.token,
                "results": ["baseline", *self.results()], "result_stats": result_stats,
                "job": job, "point_limit": MAX_POINTS, "stats": stats,
                "process_id": os.getpid(), "uptime_seconds": time.monotonic() - self.started,
                "warning": ("Nube externa: solo RGB y normales; vuelve al proyecto para usar etiquetas y mapeos." if self.custom_input else "Clases heurísticas sin anotaciones; Poisson puede inferir superficies no observadas.")}

    def results(self) -> dict[str, Path]:
        root = self.root / "outputs/app_runs"
        found = {}
        if root.is_dir():
            found = {p.parent.name: p for p in sorted(root.glob("*/labels.npz")) if (p.parent / "metrics.json").is_file()}
            found.update({p.parent.name: p for p in sorted(root.glob("*/hybrid_tiles_labels.npz")) if (p.parent / "hybrid_tiles_metrics.json").is_file()})
        semantic_root = self.root / "outputs/semantic"
        if semantic_root.is_dir():
            found.update({
                p.parent.name: p for p in sorted(semantic_root.glob("*/semantic_v3_labels.npz"))
                if (p.parent.name == "global_local_v3_release" or p.parent.name.startswith("v3_"))
                and (p.parent / "semantic_v3_original.ply").is_file()
            })
        return found

    def export_ply(self, result: str) -> Path:
        """Resolve only known, generated full-resolution semantic PLY files."""
        if result == "baseline":
            path = self.root / "outputs/semantic/global_heuristic_v1/semantic_original.ply"
        else:
            labels = self.results().get(result)
            if labels is None or labels.name != "semantic_v3_labels.npz":
                raise FileNotFoundError("No hay un PLY semántico completo para este resultado")
            path = labels.parent / "semantic_v3_original.ply"
        if not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
            raise FileNotFoundError("El PLY semántico todavía no está disponible")
        return path

    def set_input(self, value: str) -> None:
        if not isinstance(value, str): raise ValueError("La ruta debe ser texto")
        path = Path(value).expanduser().resolve() if value.strip() else None
        if path is not None and (path.suffix.lower() != ".ply" or not path.is_file()):
            raise ValueError("Indica un archivo PLY local existente")
        with self.lock:
            self.custom_input = path
            self.cached_name = None
            self.cached_cloud = None

    def cloud(self, resolution: str, mode: str, limit: int, hide_unknown: bool, result: str = "baseline") -> tuple[bytes, dict]:
        if resolution not in CLOUDS or mode not in ("rgb", "normals", "instances", "semantic"):
            raise ValueError("Resolución o vista desconocida")
        if not 1 <= limit <= MAX_POINTS:
            raise ValueError(f"Límite de puntos permitido: 1–{MAX_POINTS}")
        with self.lock:
            if self.custom_input is not None and (resolution != "original" or mode in ("instances", "semantic")):
                raise ValueError("La nube externa no tiene mapeos ni etiquetas de este proyecto")
            path = self.custom_input if self.custom_input is not None else self.path(CLOUDS[resolution])
            if not path.is_file():
                raise FileNotFoundError(f"No existe la nube {resolution}")
            if self.cached_name != resolution:
                self.cached_cloud = None
                self.cached_name = None
                self.cached_cloud = load_ply(path)
                self.cached_name = resolution
            cloud = self.cached_cloud
            xyz = np.asarray(cloud.points)
            count = len(xyz)
            ids = np.random.default_rng(42).permutation(count)[:min(count, limit)]
            points = xyz[ids]
            valid = np.isfinite(points).all(axis=1)
            ids, points = ids[valid], points[valid]
            if mode == "rgb":
                rgb = np.asarray(cloud.colors)[ids] if cloud.has_colors() else np.full((len(ids), 3), .55)
                if not cloud.has_colors():
                    raise ValueError("La nube no contiene RGB")
            elif mode == "normals":
                if not cloud.has_normals():
                    raise ValueError("La nube no contiene normales")
                rgb = normals_colors(np.asarray(cloud.normals)[ids])
            else:
                if result == "baseline":
                    path = self.path(GEOMETRY if mode == "instances" else SEMANTIC)
                else:
                    path = self.results().get(result)
                    if path is None: raise FileNotFoundError("Resultado no disponible")
                if not path.is_file():
                    raise FileNotFoundError("No existen etiquetas para esta vista")
                hybrid = result != "baseline" and path.name == "hybrid_tiles_labels.npz"
                if hybrid:
                    baseline_path = self.path(GEOMETRY if mode == "instances" else SEMANTIC)
                    with np.load(baseline_path) as data:
                        key = ("global_labels" if mode == "instances" else "semantic_labels_global") if resolution == "global" else ("original_labels" if mode == "instances" else "semantic_labels_original")
                        labels = data[key].copy()
                    if mode == "semantic":
                        with np.load(path) as data:
                            suffix = "global" if resolution == "global" else "original"
                            rows = data[suffix + "_indices"]
                            v1 = data["semantic_labels_v1_" + suffix]
                            v2 = data["semantic_labels_v2_" + suffix]
                            if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
                                    or v1.shape != rows.shape or v2.shape != rows.shape
                                    or len(np.unique(rows)) != len(rows)
                                    or np.any((rows < 0) | (rows >= len(labels)))
                                    or np.any((v2 < 0) | (v2 >= len(CLASS_IDS)))
                                    or not np.array_equal(labels[rows], v1)
                                    or not np.array_equal(v2[v1 != CLASS_IDS["unknown"]], v1[v1 != CLASS_IDS["unknown"]])):
                                raise ValueError("El ROI híbrido no coincide con la baseline")
                            labels[rows] = v2
                else:
                    with np.load(path) as data:
                        if result == "baseline":
                            key = ("global_labels" if mode == "instances" else "semantic_labels_global") if resolution == "global" else ("original_labels" if mode == "instances" else "semantic_labels_original")
                        else:
                            key = ("instance_" if mode == "instances" else "semantic_") + ("global" if resolution == "global" else "original")
                        if key not in data: raise ValueError("Resultado sin etiquetas alineadas")
                        labels = data[key].copy()
                if resolution == "detail":
                    with np.load(self.root / "outputs/processed/detail_mapping.npz") as mapping:
                        reps = mapping["voxel_to_representative"]
                    if len(reps) != count or np.any((reps < 0) | (reps >= len(labels))):
                        raise ValueError("El mapeo de detalle no coincide con las etiquetas originales")
                    sampled = labels[reps[ids]]
                else:
                    if len(labels) != count:
                        raise ValueError("Las etiquetas no coinciden con la nube")
                    sampled = labels[ids]
                if hide_unknown:
                    mask = sampled >= 0 if mode == "instances" else sampled != CLASS_IDS["unknown"]
                    points, sampled = points[mask], sampled[mask]
                rgb = label_colors(sampled) if mode == "instances" else semantic_colors(sampled)
            data = np.column_stack((points, np.clip(rgb, 0, 1))).astype("<f4", copy=False)
            return data.tobytes(), {"total": count, "shown": len(data), "mode": mode, "resolution": resolution}

    def mesh(self, name: str) -> bytes:
        path = self.meshes().get(name)
        if path is None:
            raise FileNotFoundError("Malla no disponible")
        mesh = o3d.io.read_triangle_mesh(str(path))
        triangles = np.asarray(mesh.triangles)
        if len(triangles) > MAX_TRIANGLES:
            triangles = triangles[np.random.default_rng(42).choice(len(triangles), MAX_TRIANGLES, replace=False)]
        vertices = np.asarray(mesh.vertices)
        if not len(triangles) or not np.isfinite(vertices).all():
            raise ValueError("Malla vacía o con vértices inválidos")
        return vertices[triangles].reshape(-1, 3).astype("<f4").tobytes()

    def start_job(self, payload: dict) -> dict:
        if self.custom_input is not None:
            raise ValueError("Vuelve al conjunto del proyecto para procesar con sus mapeos")
        kind = payload.get("kind")
        if kind not in ("sample_segmentation", "reconstruction", "hybrid_roi", "full_segmentation", "semantic_v3"):
            raise ValueError("Acción de procesamiento desconocida")
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("Ya hay un procesamiento activo")
            name = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            output = self.root / "outputs/app_runs" / name
            if kind == "semantic_v3":
                output = self.root / "outputs/semantic" / ("v3_" + name)
                command = [sys.executable, str(self.root / "scripts/segment_v3.py"), "--output-dir", str(output),
                           "--tile-size", "2.0", "--halo", "0.20", "--voxel-size", "0.20",
                           "--normal-angle", "35", "--max-global-points", "400000",
                           "--max-original-points", "2000000", "--max-tile-points", "5000",
                           "--max-edges", "100000", "--max-tiles", "5000", "--max-context-points", "2000000",
                           "--max-memory-mb", "1024"]
            elif kind == "sample_segmentation":
                sample = int(payload.get("sample", 3000))
                if not 100 <= sample <= 5000:
                    raise ValueError("La muestra debe estar entre 100 y 5000 puntos")
                command = [sys.executable, str(self.root / "scripts/segment_cloud.py"), "--sample-size", str(sample), "--iterations", "100", "--min-plane-points", "100", "--max-planes", "3", "--output-dir", str(output)]
            elif kind == "full_segmentation":
                command = [sys.executable, str(self.root / "scripts/process_project.py"), "--output-dir", str(output), "--timeout", "420", "--max-global-points", "400000"]
            else:
                roi = np.asarray(payload.get("roi"), dtype=float)
                if roi.shape != (6,) or not np.isfinite(roi).all() or np.any(roi[::2] >= roi[1::2]):
                    raise ValueError("ROI requiere seis límites finitos y ordenados")
                if kind == "reconstruction":
                    command = [sys.executable, str(self.root / "scripts/reconstruct_cloud.py"), "--input", str(self.path(CLOUDS["global"])), "--roi", *[str(float(v)) for v in roi], "--max-points", "3000", "--poisson-depth", "5", "--output-dir", str(output)]
                else:
                    command = [sys.executable, str(self.root / "scripts/run_hybrid_tiles.py"), "--roi", *[str(float(v)) for v in roi], "--tile-size", "1", "--halo", "0.2", "--max-tiles", "5000", "--max-roi-points", "5000", "--max-tile-points", "5000", "--max-edges", "100000", "--output-dir", str(output)]
            timeout = 1200 if kind in ("full_segmentation", "semantic_v3") else 300
            self.job_started = time.monotonic()
            self.job = {"state": "running", "message": f"Procesando; límite de {timeout} segundos", "output": str(output.relative_to(self.root))}
            threading.Thread(target=self._run_job, args=(command, output, timeout), daemon=True).start()
            return self.job.copy()

    def _run_job(self, command: list[str], output: Path, timeout: int) -> None:
        try:
            result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
            state = "complete" if result.returncode == 0 else "error"
            message = "Procesamiento terminado" if state == "complete" else (result.stderr or result.stdout)[-3000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            state, message = "error", str(exc)
        with self.lock:
            self.job = {"state": state, "message": message, "output": str(output.relative_to(self.root)), "elapsed_seconds": round(time.monotonic() - self.job_started, 1) if self.job_started is not None else None}


def create_server(project: Project, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def send_data(self, status: int, body: bytes, kind: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items(): self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, data: dict) -> None:
            self.send_data(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            try:
                url = urlsplit(self.path)
                q = parse_qs(url.query)
                if url.path == "/api/metadata":
                    return self.send_json(200, project.metadata())
                if url.path == "/api/cloud":
                    body, info = project.cloud(q.get("resolution", ["global"])[0], q.get("mode", ["rgb"])[0], int(q.get("limit", ["150000"])[0]), q.get("hide_unknown", ["0"])[0] == "1", q.get("result", ["baseline"])[0])
                    return self.send_data(200, body, "application/octet-stream", {"X-Point-Count": info["shown"], "X-Source-Count": info["total"]})
                if url.path == "/api/mesh":
                    return self.send_data(200, project.mesh(q.get("name", [""])[0]), "application/octet-stream")
                if url.path == "/api/export":
                    result = q.get("result", ["baseline"])[0]
                    path = project.export_ply(result)
                    body = path.read_bytes()
                    filename = path.name
                    return self.send_data(200, body, "application/octet-stream", {
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    })
                files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
                if url.path not in files: return self.send_json(404, {"error": "Ruta no encontrada"})
                filename, content_type = files[url.path]
                return self.send_data(200, (ASSETS / filename).read_bytes(), content_type)
            except FileNotFoundError as exc: return self.send_json(404, {"error": str(exc)})
            except (ValueError, OSError, RuntimeError, KeyError, IndexError, TypeError) as exc: return self.send_json(400, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                if self.path not in ("/api/jobs", "/api/input"): return self.send_json(404, {"error": "Ruta no encontrada"})
                if self.headers.get("X-Project-Token") != project.token: return self.send_json(403, {"error": "Token inválido"})
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096: raise ValueError("Solicitud inválida")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict): raise ValueError("La solicitud debe ser un objeto JSON")
                if self.path == "/api/input":
                    project.set_input(payload.get("path", ""))
                    return self.send_json(200, project.metadata())
                return self.send_json(202, project.start_job(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as exc: return self.send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            print("[web] " + format % args, file=sys.stderr)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
