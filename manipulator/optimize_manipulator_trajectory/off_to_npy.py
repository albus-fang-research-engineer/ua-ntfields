#!/usr/bin/env python3
"""
Convert a vertex-only .OFF file to a NumPy .npy file.

Output:
  <stem>.npy  — float32 array of shape (V, 3)

Usage:
  python off_to_npy.py mesh.off
  python off_to_npy.py mesh.off --output-dir ./output
"""

import argparse
import numpy as np
from pathlib import Path


def parse_off_vertices(filepath: str) -> np.ndarray:
    """
    Parse a vertex-only .OFF file and return vertices as a NumPy array.

    Args:
        filepath: Path to the .off file.

    Returns:
        vertices: float32 array of shape (V, 3)
    """
    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    idx = 0

    # ── Header ───────────────────────────────────────────────────────────────
    header = lines[idx]
    if not header.upper().startswith("OFF"):
        raise ValueError(f"File does not start with OFF header. Got: {header!r}")

    remainder = header[3:].strip()
    idx += 1

    count_line = remainder if remainder else lines[idx]
    if not remainder:
        idx += 1

    num_verts = int(count_line.split()[0])

    # ── Vertices ─────────────────────────────────────────────────────────────
    vertices = []
    for _ in range(num_verts):
        x, y, z = lines[idx].split()[:3]
        vertices.append([float(x), float(y), float(z)])
        idx += 1

    return np.array(vertices, dtype=np.float32)


def save_npy(filepath: str, output_dir: str | None = None) -> None:
    src = Path(filepath).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    out_dir = Path(output_dir).resolve() if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{src.stem}.npy"

    print(f"Reading  : {src}")
    vertices = parse_off_vertices(str(src))
    np.save(str(out_path), vertices)

    print(f"Vertices : {vertices.shape}  →  {out_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a vertex-only .OFF file to a NumPy .npy array."
    )
    parser.add_argument("input", help="Path to the input .off file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the .npy file (default: same as input)",
    )
    args = parser.parse_args()
    save_npy(args.input, args.output_dir)


if __name__ == "__main__":
    main()