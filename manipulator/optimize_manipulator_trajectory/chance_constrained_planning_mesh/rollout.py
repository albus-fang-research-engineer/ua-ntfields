"""
rollout_optimized.py  (mesh-inference version)

The only wiring change vs the original:
  - Build MeshDistanceModel once before the loop
  - Pass mesh_model instead of (model, device) to the solver
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# from mesh_inference.mesh_inference import MeshDistanceModel
from chance_constrained_planning_mesh.optimizer import solve_step_ur5e


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def rollout_optimized(start, path, mesh_model, epoch, folder="",verbose=False):
    """
    Parameters
    ----------
    start      : (6,) initial joint config
    path       : list of (6,) waypoints
    mesh_model : MeshDistanceModel  (built once with the fixed obstacle cloud)
    epoch      : int, for debug naming
    folder     : debug plot dir
    """
    p    = _to_numpy(start).copy()
    traj = [p.copy()]

    for idx, wp in enumerate(path):
        wp_np = _to_numpy(wp)

        dq_raw = (wp_np - p + np.pi) % (2 * np.pi) - np.pi
        if verbose:
            print("\n[ROLLOUT DEBUG]")
            print("waypoint idx:", idx)
            print("p:", p)
            print("wp:", wp_np)
            print("wrapped joint delta:", dq_raw)
            print("max abs joint delta:", np.max(np.abs(dq_raw)))
            print("norm joint delta:", np.linalg.norm(dq_raw))
        p, debug = solve_step_ur5e(
            q0         = p,
            q_goal     = wp_np,
            mesh_model = mesh_model,
            epoch      = epoch,
            folder     = folder,
        )
        p = _to_numpy(p).copy()
        traj.append(p.copy())

    return traj

    # for wp in path:
    #     wp_np = _to_numpy(wp)
    #     p, debug = solve_step_ur5e(
    #         q0         = p,
    #         q_goal     = wp_np,
    #         mesh_model = mesh_model,
    #         epoch      = epoch,
    #         folder     = folder,
    #     )
    #     p = _to_numpy(p).copy()
    #     traj.append(p.copy())

    # return traj