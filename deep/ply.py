"""Binary little-endian PLY point clouds: write and read back (#25)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {np.dtype("float32"): "float", np.dtype("uint8"): "uchar", np.dtype("int32"): "int",
          np.dtype("uint32"): "uint", np.dtype("float64"): "double", np.dtype("int16"): "short"}
_BACK = {v: k for k, v in _TYPES.items()}


def write_ply(path, X, rgb, extras=None, comments=()):
    """X (n, 3) world points, rgb (n, 3) uint8, extras {name: array} (one value per point, dtype kept)."""
    n = len(X)
    fields = [("x", np.float32), ("y", np.float32), ("z", np.float32),
              ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8)]
    extras = extras or {}
    for name, a in extras.items():
        fields.append((name, np.asarray(a).dtype))
    rec = np.empty(n, dtype=[(k, np.dtype(t).newbyteorder("<")) for k, t in fields])
    rec["x"], rec["y"], rec["z"] = X[:, 0], X[:, 1], X[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name, a in extras.items():
        rec[name] = a
    head = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments]
    head += [f"element vertex {n}"] + [f"property {_TYPES[np.dtype(t)]} {k}" for k, t in fields] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(rec.tobytes())
    return Path(path).stat().st_size


def read_ply(path):
    """The vertex records of a binary little-endian PLY written by write_ply, as a numpy structured array."""
    with open(path, "rb") as f:
        props, n = [], None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, _BACK[t].newbyteorder("<")))
            elif line == "end_header":
                break
        return np.frombuffer(f.read(), dtype=props, count=n)
