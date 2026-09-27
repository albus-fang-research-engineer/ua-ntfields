import numpy as np
import os

NPY_PATH = "/antfields/Experiments/03_18_12_10/epoch_1250_optimized.npy"
NPY_PATH = "/antfields/Experiments/03_18_09_59/epoch_1750_optimized.npy"

def print_waypoints():
    if not os.path.exists(NPY_PATH):
        print(f"[ERROR] File not found: {NPY_PATH}")
        return

    try:
        waypoints = np.load(NPY_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load numpy file: {e}")
        return

    print(f"[INFO] Loaded waypoints with shape: {waypoints.shape}\n")

    for i, wp in enumerate(waypoints):
        print(f"Waypoint {i}: {wp}")


if __name__ == "__main__":
    print_waypoints()