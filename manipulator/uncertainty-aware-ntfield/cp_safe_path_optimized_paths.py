"""
Finds all path_***.npy in paths_0cm that do NOT have a path_***_collision.npy
counterpart and copies them to optimized_paths/.
"""

import shutil
import re
from pathlib import Path

SRC_DIR  = Path("Evaluations/Arm/paths_0cm")
DEST_DIR = Path("Evaluations/Arm/optimized_paths")

DEST_DIR.mkdir(parents=True, exist_ok=True)

path_pattern = re.compile(r'^path_(\d{3})\.npy$')

copied  = []
skipped = []

for f in sorted(SRC_DIR.iterdir()):
    m = path_pattern.match(f.name)
    if not m:
        continue
    idx = m.group(1)
    collision_file = SRC_DIR / f"path_{idx}_collision.npy"
    if collision_file.exists():
        skipped.append(int(idx))
    else:
        shutil.copy2(f, DEST_DIR / f.name)
        copied.append(int(idx))

print(f"Copied  ({len(copied):3d}): {copied}")
print(f"Skipped ({len(skipped):3d}): {skipped}")
print(f"\nDone — {len(copied)} files copied to {DEST_DIR}")