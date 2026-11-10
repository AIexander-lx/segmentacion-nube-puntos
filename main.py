"""Start the local factory viewer without recomputing existing results."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visor local de la fábrica 3D")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    parser.add_argument("--port", type=int, default=0, help="Puerto local; 0 elige uno libre")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 0 y 65535")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    try:
        from segmentacion.webapp import Project, create_server
    except ImportError as exc:
        print(f"Falta una dependencia del proyecto: {exc}. Ejecuta uv sync.", file=sys.stderr)
        return 1
    project = Project()
    try:
        server = create_server(project, args.port)
    except OSError as exc:
        print(f"No se pudo iniciar el servidor local: {exc}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Fábrica 3D: {url}", flush=True)
    for name, present in project.metadata()["clouds"].items():
        if not present: print(f"Nube no disponible: {name}", flush=True)
    if not args.no_browser:
        threading.Timer(.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
