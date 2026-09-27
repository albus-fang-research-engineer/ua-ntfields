#!/usr/bin/env python3
"""
Reads a .off file, extracts all points, downsamples using a 0.03 voxel grid,
and saves the occupied voxel centers as occupied_voxel_centers.npy
"""

import sys
import numpy as np


def read_off(filepath: str) -> np.ndarray:
    """Parse an OFF file and return vertices as an (N, 3) float32 array."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Strip comments and blank lines
    lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    # First line must be "OFF" (possibly with counts on the same line)
    idx = 0
    header = lines[idx]
    if header.startswith("OFF"):
        remainder = header[3:].strip()
        idx += 1
        if remainder:
            # counts were on the same line as OFF
            counts_line = remainder
        else:
            counts_line = lines[idx]
            idx += 1
    else:
        raise ValueError(f"Not a valid OFF file (expected 'OFF' header, got '{header}')")

    parts = counts_line.split()
    num_vertices = int(parts[0])
    # parts[1] = num_faces, parts[2] = num_edges (ignored)

    # Read vertices
    vertices = []
    for i in range(num_vertices):
        coords = lines[idx + i].split()
        vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])

    return np.array(vertices, dtype=np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.03) -> np.ndarray:
    """
    Downsample a point cloud using a voxel grid.
    Each occupied voxel is represented by its center (not centroid).

    Args:
        points:     (N, 3) array of 3-D points.
        voxel_size: Side length of each cubic voxel.

    Returns:
        (M, 3) array of voxel center coordinates, one per occupied voxel.
    """
    # Compute voxel indices for every point
    indices = np.floor(points / voxel_size).astype(np.int64)

    # Keep only unique voxels
    unique_indices = np.unique(indices, axis=0)

    # Voxel center = (index + 0.5) * voxel_size
    centers = (unique_indices + 0.5) * voxel_size

    return centers.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 off_to_voxel.py <input.off> [voxel_size]")
        sys.exit(1)

    input_path = sys.argv[1]
    voxel_size = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.03
    output_path = "occupied_voxel_centers.npy"

    print(f"Reading:      {input_path}")
    points = read_off(input_path)
    print(f"  Loaded {len(points):,} vertices")

    print(f"Downsampling: voxel_size = {voxel_size}")
    centers = voxel_downsample(points, voxel_size)
    print(f"  Kept  {len(centers):,} occupied voxels")

    np.save(output_path, centers)
    print(f"Saved:        {output_path}  (shape {centers.shape}, dtype {centers.dtype})")


if __name__ == "__main__":
    main()