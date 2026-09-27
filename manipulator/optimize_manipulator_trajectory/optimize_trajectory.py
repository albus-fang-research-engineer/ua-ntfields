import numpy as np
from pathlib import Path

from chance_constrained_planning_mesh.rollout import rollout_optimized
from chance_constrained_planning_mesh.optimizer import solve_step_ur5e, print_waypoint_distances, BETA
from mesh_inference.mesh_inference import MeshDistanceModel

URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "datasets/arm/UR5/meshes/sphere/sphere"
OBS_PATH   = "occupied_voxel_centers.npy"   # update to your actual path
OBS_PATH   = "realpc_scaled_downsampled.off"
OBS_PATH   = "realpc_scaled.off"
# OBS_PATH   = ""
def load_obstacle_points(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Obstacle file not found: {file_path}")
    
    if file_path.suffix == ".npy":
        return np.load(file_path, allow_pickle=True)
    
    elif file_path.suffix == ".off":
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # OFF header - first line is "OFF", second is "n_verts n_faces n_edges"
        start = 0
        if lines[0].strip() == "OFF":
            start = 1
        
        n_verts, n_faces, _ = map(int, lines[start].strip().split())
        
        vertices = []
        for line in lines[start + 1 : start + 1 + n_verts]:
            vertices.append(list(map(float, line.strip().split())))
        
        return np.array(vertices, dtype=np.float32)  # (M, 3)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
def load_trajectory(file_path):
    """
    Load joint space trajectory from .npy file.

    Args:
        file_path (str or Path): Path to the .npy file

    Returns:
        np.ndarray: trajectory array of shape (T, 6)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path, allow_pickle=True)


def load_obstacle_points_simple(file_path):
    """
    Load obstacle point cloud from .npy file.

    Args:
        file_path (str or Path): Path to the .npy file  shape (M, 3)

    Returns:
        np.ndarray: obstacle points
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Obstacle file not found: {file_path}")
    return np.load(file_path, allow_pickle=True)


def main():
    # ── Load nominal trajectory ────────────────────────────────────────────
    traj_path = "Evaluations/Arm/ur5_joint_trajectory.npy"
    traj_path = "Evaluations/Arm/ur5_joint_trajectory_dense_pointcloud.npy"
    traj_path = "Evaluations/Arm/paths_0cm/path_003_collision.npy"
    traj_path = "ur5_joint_trajectory_demo.npy"
    # traj_path = "Evaluations/Arm/paths_0cm/path_004_collision.npy"
    traj = load_trajectory(traj_path)

    print("Trajectory loaded successfully!")
    print(f"  Type : {type(traj)}")
    print(f"  Shape: {traj.shape}")
    print(f"  Dtype: {traj.dtype}")
    print("\nFirst 5 timesteps:")
    print(traj[:5])

    # ── Load obstacle point cloud and build mesh model (once) ──────────────
    obstacle_points = load_obstacle_points(OBS_PATH)
    print(f"\nObstacle cloud shape: {obstacle_points.shape}")

    device     = "cuda"
    mesh_model = MeshDistanceModel(
        urdf_path       = URDF_PATH,
        sphere_dir      = SPHERE_DIR,
        obstacle_points = obstacle_points,
        end_link        = "wrist_3_link",
        device          = device,
    )
    print("MeshDistanceModel built.")
    print("\nNominal trajectory distances:")
    # print_waypoint_distances(traj, mesh_model, beta=BETA)
    # ── Run chance-constrained rollout ─────────────────────────────────────
    start     = traj[0]        # (6,) initial config
    waypoints = traj[1:]       # remaining configs as waypoints

    optimized_traj = rollout_optimized(
        start      = start,
        path       = waypoints,
        mesh_model = mesh_model,
        epoch      = 0,
        folder     = "",       # set to a path string to enable debug plots
    )

    # ── Save optimized trajectory ──────────────────────────────────────────
    save_dir = Path("joint_paths")
    save_dir.mkdir(parents=True, exist_ok=True)

    # save_path = save_dir / "optimized_joint_trajectory_0cm_path003_tuning.npy"
    save_path = save_dir / "optimized_joint_trajectory_demo.npy"
    np.save(save_path, np.asarray(optimized_traj))
    optimized_traj = np.asarray(optimized_traj)

    print("\nOptimized trajectory distances:")
    # print_waypoint_distances(optimized_traj, mesh_model, beta=BETA)
    print(f"\nOptimized trajectory saved to: {save_path}")

    print(f"\nOptimized trajectory length: {len(optimized_traj)}")
    print("First 5 optimized configs:")
    for i, q in enumerate(optimized_traj[:5]):
        print(f"  step {i}: {np.round(q, 4)}")


if __name__ == "__main__":
    main()