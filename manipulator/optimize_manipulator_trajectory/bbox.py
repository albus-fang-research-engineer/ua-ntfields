#!/usr/bin/env python3
"""
Compute and print the axis-aligned bounding box of a point cloud
stored in either Wavefront .obj or .off format.

Usage:
    python bbox.py model.obj
    python bbox.py model.off
    cat model.off | python bbox.py --format off
"""

import argparse
import sys


def parse_obj(lines):
    """Yield (x, y, z) tuples from vertex ('v') lines of an OBJ file."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            yield (float(parts[1]), float(parts[2]), float(parts[3]))


def parse_off(lines):
    """Yield (x, y, z) tuples from the vertex block of an OFF file."""
    # Strip comments and blank lines first.
    tokens = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.append(line)

    if not tokens:
        return

    # The header keyword may be on its own line ("OFF") or the counts may be
    # appended to it ("OFF 100 0 0").
    header = tokens[0].split()
    idx = 0
    if header[0].upper().endswith("OFF"):
        if len(header) >= 4:
            # Counts share the header line.
            num_vertices = int(header[1])
            idx = 1
        else:
            # Counts are on the next line.
            counts = tokens[1].split()
            num_vertices = int(counts[0])
            idx = 2
    else:
        # No explicit OFF keyword; assume first line holds the counts.
        num_vertices = int(header[0])
        idx = 1

    for line in tokens[idx:idx + num_vertices]:
        parts = line.split()
        if len(parts) >= 3:
            yield (float(parts[0]), float(parts[1]), float(parts[2]))


def bounding_box(points):
    """Return ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    count = 0
    for p in points:
        count += 1
        for i in range(3):
            if p[i] < mins[i]:
                mins[i] = p[i]
            if p[i] > maxs[i]:
                maxs[i] = p[i]
    if count == 0:
        raise ValueError("No vertices found in the input.")
    return tuple(mins), tuple(maxs), count


def detect_format(path, override):
    if override:
        return override.lower()
    lower = path.lower()
    if lower.endswith(".obj"):
        return "obj"
    if lower.endswith(".off"):
        return "off"
    raise ValueError(
        "Could not infer format from extension; use --format obj|off."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Print the axis-aligned bounding box of a point cloud."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to the .obj or .off file (omit to read from stdin).",
    )
    parser.add_argument(
        "--format",
        choices=["obj", "off"],
        help="Force the input format (required when reading from stdin "
             "or when the extension is missing).",
    )
    args = parser.parse_args()

    if args.path:
        fmt = detect_format(args.path, args.format)
        with open(args.path, "r") as f:
            lines = f.readlines()
    else:
        if not args.format:
            parser.error("--format is required when reading from stdin.")
        fmt = args.format
        lines = sys.stdin.readlines()

    points = parse_obj(lines) if fmt == "obj" else parse_off(lines)
    mins, maxs, count = bounding_box(points)
    size = tuple(maxs[i] - mins[i] for i in range(3))
    center = tuple((maxs[i] + mins[i]) / 2.0 for i in range(3))

    print(f"Vertices read : {count}")
    print(f"Min corner    : x={mins[0]:.6g}  y={mins[1]:.6g}  z={mins[2]:.6g}")
    print(f"Max corner    : x={maxs[0]:.6g}  y={maxs[1]:.6g}  z={maxs[2]:.6g}")
    print(f"Size (W,H,D)  : x={size[0]:.6g}  y={size[1]:.6g}  z={size[2]:.6g}")
    print(f"Center        : x={center[0]:.6g}  y={center[1]:.6g}  z={center[2]:.6g}")
    print(f"{mins[0]:.6g} {mins[1]:.6g} {mins[2]:.6g}")
    print(f"{maxs[0]:.6g} {maxs[1]:.6g} {maxs[2]:.6g}")


if __name__ == "__main__":
    main()