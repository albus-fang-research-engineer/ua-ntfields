"""
Load pre-optimized UR5 trajectories and run a per-waypoint collision check
using the same FK + KD-tree approach as the path generation script.

Input  : Evaluations/Arm/optimized_paths/path_000.npy … path_099.npy
         Each file is an (T', 6) array in joint-space radians.

Output : Console report of collision status per trajectory + summary stats.
"""

import sys, os, math
import numpy as np
import torch
import igl
import pytorch_kinematics as pk
import torch_kdtree
from pathlib import Path

sys.path.append('.')

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
IN_DIR     = Path("Evaluations/Arm/optimized_paths_working")
IN_DIR     = Path("Evaluations/Arm/optimized_paths_working_variable_sigma")
MESH_FILE  = "./datasets/arm/UR5/realpc_scaled.off"
URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "datasets/arm/UR5/meshes/sphere/sphere"
NUM_PATHS  = 100

SCALE             = math.pi / 0.5   # must match path generation script
COLLISION_THRESHOLD = 0.0           # signed distance <= this is a collision


# ------------------------------------------------------------------ #
# Build FK chain + bounding-sphere mesh list  (identical to gen script)
# ------------------------------------------------------------------ #
def build_chain():
    end_link = "wrist_3_link"
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    chain = pk.build_serial_chain_from_urdf(
        open(URDF_PATH).read(), end_link)
    chain = chain.to(dtype=torch.float32, device=device)

    dummy = torch.rand(1, 6, device=device)
    tg    = chain.forward_kinematics(dummy, end_only=False)

    mesh_list = []
    for idx, link_name in enumerate(tg):
        if 2 <= idx < 8:
            arr = np.load(f"{SPHERE_DIR}/{link_name}.npy")
            mesh_list.append(torch.tensor(arr, dtype=torch.float32, device=device))

    return chain, mesh_list


# ------------------------------------------------------------------ #
# FK → sphere centers in world frame  (identical to gen script)
# ------------------------------------------------------------------ #
def FK_spheres(joint_angles_rad: torch.Tensor,
               chain, mesh_list) -> torch.Tensor:
    """
    joint_angles_rad : (N, 6)  raw radians
    Returns          : (N, M, 4)  sphere centers (xyz) + radii
    """
    tg_batch = chain.forward_kinematics(joint_angles_rad, end_only=False)
    p_list   = []
    for idx, link_name in enumerate(tg_batch):
        if 2 <= idx < 8:
            balls = mesh_list[idx - 2]
            ones  = torch.ones(balls.shape[0], 1, device="cuda")
            nv    = torch.cat([balls[:, :3], ones], dim=1)
            m     = tg_batch[link_name].get_matrix()
            p     = torch.bmm(m, nv.T.unsqueeze(0).expand(m.shape[0], -1, -1))
            p     = p.permute(0, 2, 1)
            p[..., 3] = balls[:, 3]
            p_list.append(p)
    return torch.cat(p_list, dim=1)


# ------------------------------------------------------------------ #
# Signed distance for a batch of configs  (identical to gen script)
# ------------------------------------------------------------------ #
def arm_collision_distance(joint_batch_rad: torch.Tensor,
                           chain, mesh_list, kdtree) -> torch.Tensor:
    """
    joint_batch_rad : (N, 6) in raw radians
    Returns         : (N,)  minimum signed distance — positive = free
    """
    spheres      = FK_spheres(joint_batch_rad, chain, mesh_list)   # (N, S, 4)
    N, S, _      = spheres.shape
    flat         = spheres.reshape(-1, 4)

    dists, _     = kdtree.query(flat[:, :3], nr_nns_searches=1)
    dists        = dists.squeeze()
    signed       = torch.sqrt(dists) - flat[:, 3]

    signed       = signed.reshape(N, S)
    min_dist, _  = signed.min(dim=1)
    return min_dist


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Building FK chain ...")
    chain, mesh_list = build_chain()

    print("=== Loading obstacle mesh ...")
    v, _ = igl.read_triangle_mesh(MESH_FILE)
    v_obs  = torch.tensor(v, dtype=torch.float32, device=device)
    kdtree = torch_kdtree.build_kd_tree(v_obs)
    print(f"    Obstacle mesh vertices: {v_obs.shape[0]}\n")

    collision_free_count = 0
    checked_count        = 0
    skipped_count        = 0
    min_distances        = []

    for i in range(NUM_PATHS):
        traj_path = IN_DIR / f"path_{i:03d}.npy"

        if not traj_path.exists():
            print(f"[{i+1:03d}/{NUM_PATHS}]  MISSING {traj_path} — skipping")
            skipped_count += 1
            continue

        traj_rad = np.load(traj_path, allow_pickle=True)   # (T, 6) radians
        traj_rad = np.asarray(traj_rad)
        print(f"[{i+1:03d}/{NUM_PATHS}]  shape={traj_rad.shape}", end="  ")

        waypoints = torch.tensor(traj_rad, dtype=torch.float32, device=device)

        dist         = arm_collision_distance(waypoints, chain, mesh_list, kdtree)
        in_collision = dist <= COLLISION_THRESHOLD 

        n_col        = int(in_collision.sum().item())
        min_dist     = float(dist.min().item())
        min_distances.append(min_dist)
        checked_count += 1

        if n_col == 0:
            collision_free_count += 1
            print(f"COLLISION-FREE   min_dist={min_dist:.4f}")
        else:
            col_indices = in_collision.nonzero(as_tuple=True)[0].tolist()
            print(
                f"COLLISION        min_dist={min_dist:.4f}  "
                f"colliding waypoints={col_indices}"
            )

    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print(f"Trajectories checked    : {checked_count}/{NUM_PATHS}")
    print(f"Skipped (missing file)  : {skipped_count}/{NUM_PATHS}")
    print(f"Collision-free          : {collision_free_count}/{checked_count}")
    print(f"In collision            : {checked_count - collision_free_count}/{checked_count}")
    if min_distances:
        print(f"Min distance across all : {min(min_distances):.4f}")
        print(f"Mean min distance       : {np.mean(min_distances):.4f}")


if __name__ == "__main__":
    main()