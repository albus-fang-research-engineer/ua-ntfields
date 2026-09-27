"""
Optimize all 100 randomly-generated UR5 trajectories using the
chance-constrained optimizer and save each result as an individual .npy file.

Input  : Evaluations/Arm/paths/path_000.npy … path_099.npy  (T, 6) rad
Output : Evaluations/Arm/optimized_paths/path_000.npy … path_099.npy
         Each file is an (T', 6) array in joint-space radians.
"""

import numpy as np
from pathlib import Path

from chance_constrained_planning_mesh.rollout import rollout_optimized
from chance_constrained_planning_mesh.optimizer import (
    solve_step_ur5e, print_waypoint_distances, BETA
)
from mesh_inference.mesh_inference import MeshDistanceModel
import igl
# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
TRAJ_DIR   = Path("Evaluations/Arm/paths_0cm")
# OUT_DIR    = Path("Evaluations/Arm/optimized_paths")
OUT_DIR    = Path("Evaluations/Arm/optimized_tuning")
OBS_PATH   = "./datasets/arm/UR5/realpc_scaled_downsampled.off"
OBS_PATH   = "realpc_scaled_downsampled.off"
URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "datasets/arm/UR5/meshes/sphere/sphere"
NUM_PATHS  = 100


# ------------------------------------------------------------------ #
# Build the shared mesh model once
# ------------------------------------------------------------------ #
def build_mesh_model(obs_path: str, device: str = "cuda") -> MeshDistanceModel:
    
    v, _ = igl.read_triangle_mesh(obs_path)          # load .off mesh
    obstacle_points = v.astype(np.float32)            # (N, 3) vertex array
    print(f"Obstacle cloud shape: {obstacle_points.shape}")
    model = MeshDistanceModel(
        urdf_path       = URDF_PATH,
        sphere_dir      = SPHERE_DIR,
        obstacle_points = obstacle_points,
        end_link        = "wrist_3_link",
        device          = device,
    )
    print("MeshDistanceModel built.\n")
    return model


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    device     = "cuda"
    mesh_model = build_mesh_model(OBS_PATH, device=device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_count  = 0
    failed_count = 0

    for i in range(NUM_PATHS):
        traj_path = TRAJ_DIR / f"path_{i:03d}_collision.npy"
        # traj_path = TRAJ_DIR / f"path_{i:03d}.npy"
        if not traj_path.exists():
            print(f"[{i+1:03d}/{NUM_PATHS}]  MISSING {traj_path} — skipping")
            continue

        traj = np.load(traj_path, allow_pickle=True)   # (T, 6) radians
        print(f"[{i+1:03d}/{NUM_PATHS}]  {traj_path.name}  shape={traj.shape}")

        start     = traj[0]    # (6,)
        waypoints = traj[1:]   # (T-1, 6)

        try:
            optimized_traj = rollout_optimized(
                start      = start,
                path       = waypoints,
                mesh_model = mesh_model,
                epoch      = 0,
                folder     = "",
            )
            optimized_traj = np.asarray(optimized_traj)
            print(f"           optimized shape={optimized_traj.shape}")
        except Exception as e:
            print(f"           ERROR during optimization: {e}")
            optimized_traj = traj   # fall back to the original trajectory
            failed_count  += 1

        save_path = OUT_DIR / f"path_{i:03d}.npy"
        np.save(save_path, optimized_traj)
        print(f"           saved → {save_path}")
        saved_count += 1

    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print(f"\nSaved  : {saved_count}/{NUM_PATHS}  →  {OUT_DIR}")
    print(f"Skipped: {NUM_PATHS - saved_count}/{NUM_PATHS}  (missing input files)")
    print(f"Fell back to original (optimizer error): {failed_count}")


if __name__ == "__main__":
    main()