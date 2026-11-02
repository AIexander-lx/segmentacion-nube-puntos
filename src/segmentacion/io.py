"""Lectura y escritura de nubes de puntos PLY binarios (posición, color RGB y normales).

Se implementa con NumPy para no depender de Open3D en el preprocesamiento.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class PointCloud:
    """Nube de puntos: xyz (n,3) float32, rgb (n,3) uint8 y normales (n,3) float32."""

    xyz: np.ndarray
    rgb: np.ndarray
    normals: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, idx: np.ndarray) -> "PointCloud":
        return PointCloud(self.xyz[idx], self.rgb[idx], self.normals[idx])


def read_ply(path: str | Path) -> PointCloud:
    """Lee un PLY binario little-endian con x,y,z, red,green,blue y nx,ny,nz."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end += 2 if raw[end:end + 2] == b"\r\n" else 1
    header = raw[:end].decode("latin1").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Solo se admite PLY binary_little_endian")
    n, fields = 0, []
    for line in header:
        p = line.split()
        if p[:2] == ["element", "vertex"]:
            n = int(p[2])
        elif p[:1] == ["property"] and len(p) == 3 and p[1] in _PLY_TYPES:
            fields.append((p[2], "<" + _PLY_TYPES[p[1]]))
    data = np.frombuffer(raw, dtype=np.dtype(fields), count=n, offset=end)
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(np.float32)
    rgb = np.c_[data["red"], data["green"], data["blue"]].astype(np.uint8)
    nrm = np.c_[data["nx"], data["ny"], data["nz"]].astype(np.float32)
    return PointCloud(xyz, rgb, nrm)


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Escribe un PLY binario con posición y color (uint8) por punto."""
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
