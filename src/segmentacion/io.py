"""Lectura y escritura segura de nubes de puntos PLY mediante Open3D."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import open3d as o3d

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class PointCloud:
    """Nube ligera para el preprocesamiento NumPy: XYZ, RGB y normales."""

    xyz: np.ndarray
    rgb: np.ndarray
    normals: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, idx: np.ndarray) -> "PointCloud":
        return PointCloud(self.xyz[idx], self.rgb[idx], self.normals[idx])


def read_ply(path: str | Path) -> PointCloud:
    """Lee PLY binario little-endian con XYZ, RGB y normales sin Open3D."""
    raw = Path(path).read_bytes()
    header_end = raw.index(b"end_header") + len(b"end_header")
    header_end += 2 if raw[header_end:header_end + 2] == b"\r\n" else 1
    header = raw[:header_end].decode("latin1").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Solo se admite PLY binary_little_endian")

    count, fields = 0, []
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[:1] == ["property"] and len(parts) == 3 and parts[1] in _PLY_TYPES:
            fields.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
    data = np.frombuffer(raw, dtype=np.dtype(fields), count=count, offset=header_end)
    xyz = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    rgb = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    normals = np.column_stack((data["nx"], data["ny"], data["nz"])).astype(np.float32)
    return PointCloud(xyz, rgb, normals)


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Escribe un PLY binario little-endian con coordenadas y color RGB."""
    xyz, rgb = np.asarray(xyz), np.asarray(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("xyz y rgb deben tener la misma forma (n, 3)")
    if not np.isfinite(xyz).all() or np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("xyz debe ser finito y rgb debe estar en [0, 255]")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header.encode("ascii") + data.tobytes())


def load_ply(path: str | Path) -> o3d.geometry.PointCloud:
    """Carga un PLY existente y rechaza archivos vacíos o ilegibles."""
    import open3d as o3d

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el archivo PLY: {source}")
    pcd = o3d.io.read_point_cloud(str(source))
    if not pcd.has_points():
        raise ValueError(f"El PLY no contiene puntos legibles: {source}")
    return pcd


def save_labeled_ply(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    palette: np.ndarray,
    path: str | Path,
) -> Path:
    """Exporta una copia coloreada; nunca escribe sobre la nube de entrada."""
    import open3d as o3d

    target = Path(path).expanduser()
    if pcd.is_empty():
        raise ValueError("No se puede exportar una nube vacía")
    labels = np.asarray(labels)
    palette = np.asarray(palette, dtype=float)
    if labels.ndim != 1 or len(labels) != len(pcd.points):
        raise ValueError("labels debe tener una etiqueta por punto")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette debe tener forma (k, 3), con k > 0")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError("labels debe contener enteros >= -1")
    if not np.isfinite(palette).all() or np.any((palette < 0) | (palette > 1)):
        raise ValueError("palette debe ser finita y estar en [0, 1]")
    colors = np.full((len(labels), 3), 0.5)
    assigned = labels >= 0
    colors[assigned] = palette[labels[assigned] % len(palette)]
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() == Path.cwd().resolve() or (target.exists() and target.resolve() == Path.cwd().resolve()):
        raise ValueError("La ruta de salida debe ser un archivo, no un directorio")
    if not o3d.io.write_point_cloud(str(target), out):
        raise OSError(f"No se pudo escribir el PLY: {target}")
    return target
