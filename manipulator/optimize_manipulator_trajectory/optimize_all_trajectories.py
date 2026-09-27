"""
Optimize all 100 randomly-generated UR5 trajectories using the
chance-constrained optimizer and save each result as an individual .npy file.

Input  : Evaluations/Arm/paths/path_000.npy … path_099.npy  (T, 6) rad
Output : Evaluations/Arm/optimized_paths/path_000.npy … path_099.npy
         Each file is an (T', 6) array in joint-space radians.
"""
import cProfile, pstats, io
import numpy as np
from chance_constrained_planning_mesh.rollout import rollout_optimized
from chance_constrained_planning_mesh.optimizer import (
    solve_step_ur5e, print_waypoint_distances, BETA
)
from mesh_inference.mesh_inference import MeshDistanceModel
import igl
import time
from pathlib import Path
from datetime import datetime

PROFILE_DIR = Path("./time_profiles")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
TRAJ_DIR   = Path("Evaluations/Arm/paths_ours")
OUT_DIR    = Path("Evaluations/Arm/optimized_paths_working_variable_sigma")
OBS_PATH   = "./datasets/arm/UR5/realpc_scaled_downsampled.off"
OBS_PATH   = "realpc_scaled_downsampled.off"
URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "datasets/arm/UR5/meshes/sphere/sphere"
NUM_PATHS  = 100
EE_IDX = 5

def next_run_dir(base: Path) -> Path:
    """Create the next run_NNN/ folder under `base` and return its path."""
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base.glob("run_*"):
        if p.is_dir() and p.name[4:].isdigit():
            existing.append(int(p.name[4:]))
    n = (max(existing) + 1) if existing else 0
    run_dir = base / f"run_{n:03d}"
    run_dir.mkdir()
    return run_dir
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
    RUN_DIR = next_run_dir(PROFILE_DIR)
    print(f"Profile run dir: {RUN_DIR}\n")
    # Load success flags — shape (100,) bool
    successes = np.load(TRAJ_DIR / "successes.npy", allow_pickle=True).astype(bool)
    print(f"Successful trajectories: {successes.sum()}/{len(successes)}\n")

    saved_count  = 0
    failed_count = 0
    skipped_count = 0
    times = []
    for i in range(NUM_PATHS):
        traj_path      = TRAJ_DIR / f"path_{i:03d}.npy"
        collision_path = TRAJ_DIR / f"path_{i:03d}_collision.npy"

        # Skip if flagged as failed OR if a collision sibling file exists
        # if not successes[i] or collision_path.exists():
        if not successes[i]:
            print(f"[{i+1:03d}/{NUM_PATHS}]  SKIP  (collision or failed flag)")
            skipped_count += 1
            continue

        if not traj_path.exists():
            print(f"[{i+1:03d}/{NUM_PATHS}]  MISSING {traj_path} — skipping")
            skipped_count += 1
            continue

        traj = np.load(traj_path, allow_pickle=True)   # (T, 6) radians
        print(f"[{i+1:03d}/{NUM_PATHS}]  {traj_path.name}  shape={traj.shape}")

        start     = traj[0]    # (6,)
        waypoints = traj[1:]   # (T-1, 6)

        try:
            
            t0 = time.perf_counter()
            pr = cProfile.Profile()
            pr.enable()
            optimized_traj = rollout_optimized(
                start      = start,
                path       = waypoints,
                mesh_model = mesh_model,
                epoch      = 0,
                folder     = "",
            )
            pr.disable()
            dt = time.perf_counter() - t0
            

            # s = io.StringIO()
            # pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
            # print(s.getvalue())
            # stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # prof_path = PROFILE_DIR / f"profile_{stamp}_path{i:03d}.prof"
            # pr.dump_stats(str(prof_path))

            # # optional: also write a human-readable text summary next to it
            # with open(prof_path.with_suffix(".txt"), "w") as f:
            #     pstats.Stats(pr, stream=f).sort_stats("cumulative").print_stats(40)
            path_prof_dir = RUN_DIR / f"path_{i:03d}"
            path_prof_dir.mkdir(exist_ok=True)
            prof_path = path_prof_dir / "profile.prof"
            pr.dump_stats(str(prof_path))
            with open(prof_path.with_suffix(".txt"), "w") as f:
                pstats.Stats(pr, stream=f).sort_stats("cumulative").print_stats(40)
            n_steps = len(waypoints)
            times.append((i, dt))
            optimized_traj = np.asarray(optimized_traj)
            # print(f"           optimized shape={optimized_traj.shape}")
            print(f"           optimized shape={optimized_traj.shape}  "
                    f"time={dt:.3f}s  ({dt / max(n_steps, 1) * 1000:.1f} ms/step)")

        except Exception as e:
            print(f"           ERROR during optimization: {e}")
            optimized_traj = traj.copy()   # fall back to the original trajectory
            failed_count  += 1
        # optimized_traj[:, EE_IDX] = start[EE_IDX]
        save_path = OUT_DIR / f"path_{i:03d}.npy"
        np.save(save_path, optimized_traj)
        print(f"           saved → {save_path}")
        saved_count += 1

    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print(f"\nSaved   : {saved_count}/{NUM_PATHS}  →  {OUT_DIR}")
    print(f"Skipped : {skipped_count}/{NUM_PATHS}  (collision / failed flag / missing)")
    print(f"Fell back to original (optimizer error): {failed_count}")
    if times:
        ts = np.array([t for _, t in times])
        print(f"\nTiming over {len(ts)} optimized trajectories:")
        print(f"  total    : {ts.sum():.2f} s")
        # print(f"  per traj : mean {ts.mean():.3f} s  median {np.median(ts):.3f} s  "
        #     f"min {ts.min():.3f} s  max {ts.max():.3f} s")
        print(f"  per traj : mean {ts.mean():.3f} s  std {ts.std():.3f} s  "
                f"median {np.median(ts):.3f} s  min {ts.min():.3f} s  max {ts.max():.3f} s")
        times_arr = np.array(times, dtype=float)  # columns: [path_index, time_sec]
        time_save_path = OUT_DIR / "optimization_times_sped_up.npy"
        np.save(time_save_path, times_arr)
        # Print largest optimization times
        top_k = min(10, len(times_arr))
        largest_idx = np.argsort(times_arr[:, 1])[::-1][:top_k]

        print(f"\nTop {top_k} slowest optimized trajectories:")
        for rank, idx in enumerate(largest_idx, start=1):
            path_idx = int(times_arr[idx, 0])
            dt = times_arr[idx, 1]
            print(f"  {rank:02d}. path_{path_idx:03d}.npy  index={path_idx:03d}  time={dt:.3f}s")

if __name__ == "__main__":
    main()