"""M3's own coloured point cloud: binary little-endian PLY, float x y z (world metres) and uchar red green
blue (numpy only). The format follows tools/greedy_foveal/run.py:ply_bytes; that driver is the
playground's and is neither imported nor edited (CLAUDE.md rule 7).

Colours: the look's observed linear RGB, one fixed exposure (0.9 / the 99th luminance percentile of the
first look, as motion.visuals), sRGB-encoded for display. The linear values stay in the companion NPZ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])
_DTYPE = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]


def exposure_of(rgb_linear) -> float:
    lum = np.asarray(rgb_linear, np.float64) @ LUMA
    return float(0.9 / max(np.quantile(lum, 0.99), 1e-6))


def srgb_u8(rgb_linear, exposure: float) -> np.ndarray:
    x = np.clip(np.asarray(rgb_linear, np.float64) * exposure, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


def write_ply(path, xyz, rgb_u8, comments=()) -> Path:
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb_u8 = np.asarray(rgb_u8, np.uint8).reshape(-1, 3)
    if len(xyz) != len(rgb_u8) or not np.isfinite(xyz).all():
        raise ValueError("PLY: xyz and colours must match in length and xyz must be finite")
    rec = np.zeros(len(xyz), dtype=_DTYPE)
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        rec[k] = rgb_u8[:, i]
    head = ["ply", "format binary_little_endian 1.0", *[f"comment {c}" for c in comments],
            f"element vertex {len(xyz)}", "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(("\n".join(head) + "\n").encode() + rec.tobytes())
    return p


def read_ply(path) -> tuple[np.ndarray, np.ndarray, list]:
    """An independent reader of exactly this format (the readback check): (xyz, rgb_u8, comments)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode().splitlines()
    if lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
        raise ValueError("not a binary little-endian PLY")
    n = int(next(ln.split()[2] for ln in lines if ln.startswith("element vertex")))
    props = [ln.split()[1:] for ln in lines if ln.startswith("property")]
    if props != [["float", "x"], ["float", "y"], ["float", "z"], ["uchar", "red"], ["uchar", "green"], ["uchar", "blue"]]:
        raise ValueError(f"unexpected properties {props}")
    rec = np.frombuffer(raw[end:], dtype=_DTYPE, count=n)
    if len(raw) - end != n * 15:
        raise ValueError("vertex block length disagrees with the header")
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], -1)
    rgb = np.stack([rec["red"], rec["green"], rec["blue"]], -1)
    return xyz, rgb, [ln[len("comment "):] for ln in lines if ln.startswith("comment ")]
