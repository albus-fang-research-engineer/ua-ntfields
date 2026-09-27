#!/usr/bin/env python3
"""
Reads a vertices-only .off file and downsamples the point cloud
to a 0.03 m voxel grid, then writes the result to a new .off file.
"""

import sys
import numpy as np
from pathlib import Path


def read_off(filepath: str) -> np.ndarray:
    """
    Parse a vertices-only OFF file and return an (N, 3) float array.
    Supports both the standard header format and the compact 'OFF N 0 0' format.
    """
    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if not lines:
        raise ValueError("File is empty.")

    idx = 0

    # ── Header line ──────────────────────────────────────────────────────────
    header = lines[idx]
    idx += 1

    if header.upper().startswith("OFF"):
        remainder = header[3:].strip()
        if remainder:
            # Compact form: "OFF N 0 0"
            parts = remainder.split()
            n_verts = int(parts[0])
        else:
            # Standard form: next line holds "N 0 0"
            parts = lines[idx].split()
            idx += 1
            n_verts = int(parts[0])
    else:
        raise ValueError(f"Not a valid OFF file — unexpected header: {header!r}")

    # ── Read vertex coordinates ──────────────────────────────────────────────
    verts = []
    for _ in range(n_verts):
        coords = lines[idx].split()
        idx += 1
        verts.append([float(coords[0]), float(coords[1]), float(coords[2])])

    return np.array(verts, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.03) -> np.ndarray:
    """
    Voxel-grid downsampling: one representative point (centroid) per voxel.

    Parameters
    ----------
    points     : (N, 3) array of XYZ coordinates
    voxel_size : edge length of each voxel in metres (default 0.03)

    Returns
    -------
    (M, 3) array of downsampled points, M <= N
    """
    # Shift so minimum is at origin, then assign each point to a voxel index
    mins = points.min(axis=0)
    indices = np.floor((points - mins) / voxel_size).astype(np.int64)

    # Build a unique key per voxel using a dict  {(ix, iy, iz): [point indices]}
    voxel_map: dict = {}
    for pt_idx, key in enumerate(map(tuple, indices)):
        voxel_map.setdefault(key, []).append(pt_idx)

    # Compute the centroid for each occupied voxel
    downsampled = np.array(
        [points[idx_list].mean(axis=0) for idx_list in voxel_map.values()],
        dtype=np.float64,
    )
    return downsampled


def write_off(filepath: str, points: np.ndarray) -> None:
    """Write a vertices-only OFF file."""
    n = len(points)
    with open(filepath, "w") as f:
        f.write("OFF\n")
        f.write(f"{n} 0 0\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 voxel_downsample.py <input.off> [output.off] [voxel_size]")
        print("  output.off  defaults to <input>_downsampled.off")
        print("  voxel_size  defaults to 0.03 (metres)")
        sys.exit(1)

    input_path = sys.argv[1]

    output_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else str(Path(input_path).stem) + "_downsampled.off"
    )

    voxel_size = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.03

    print(f"Reading  : {input_path}")
    points = read_off(input_path)
    print(f"  Loaded   {len(points):,} points")

    print(f"Voxel size: {voxel_size} m")
    ds = voxel_downsample(points, voxel_size)
    print(f"  Retained {len(ds):,} points  ({len(ds)/len(points)*100:.1f} %)")

    print(f"Writing  : {output_path}")
    write_off(output_path, ds)
    print("Done.")


if __name__ == "__main__":
    main()