"""
mesh_inference.py

Drop-in replacement for mu_sigma_grad_nn using sphere-approximated robot
geometry + a KD-tree over the obstacle point cloud.

Shape contract (matches mu_sigma_grad_nn):
    mu   : (N_LINKS,) - min signed distance per link  [m]
    sigma: (N_LINKS,) - constant 0.02 m
    grad : (N_LINKS, 6) - d(min_dist_link)/dq  in joint space
    obs_k: (N_LINKS,) - index of the closest obstacle point per link (debug)

Usage
-----
    # Build once at startup - obstacle_points is fixed for the whole session:
    from mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh
    mesh_model = MeshDistanceModel(
        urdf_path       = "datasets/arm/UR5/ur5e.urdf",
        sphere_dir      = "datasets/arm/UR5/meshes/sphere/sphere",
        obstacle_points = obs_array,   # (M,3) numpy or cuda tensor - fixed
        end_link        = "wrist_3_link",
        device          = "cuda",
    )

    # Each planning step:
    #   mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q0, mesh_model)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import pytorch_kinematics as pk
import torch_kdtree

# Fixed noise level - 1 cm
SIGMA_CONST = 0.01
N_NEIGHBORS    = 10    # torch_kdtree will round up to 16
GRAD_THRESHOLD = 0.10
# Links whose spheres are loaded (matches iter>1 and iter<8 in the original FK)
_LINK_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
N_LINKS = len(_LINK_NAMES)


class MeshDistanceModel:
    """
    All heavy structures (chain, spheres, KD-tree) built once at construction.

    Parameters
    ----------
    urdf_path       : path to ur5e.urdf
    sphere_dir      : directory with <link_name>.npy files  (K,4): [cx,cy,cz,r]
    obstacle_points : (M,3) point cloud - assumed FIXED for the session
    end_link        : terminal link name for the serial chain
    device          : "cuda" or "cpu"
    """

    def __init__(
        self,
        urdf_path,
        sphere_dir,
        obstacle_points,
        end_link="wrist_3_link",
        device="cuda",
    ):
        self.device = device
        self.dtype  = torch.float32

        # -- Kinematic chain -----------------------------------------------
        with open(urdf_path) as f:
            urdf_str = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_str, end_link)
        self.chain = self.chain.to(dtype=self.dtype, device=device)

        # -- Sphere approximations -----------------------------------------
        self.sphere_tensors = []
        for name in _LINK_NAMES:
            npy_path = os.path.join(sphere_dir, f"{name}.npy")
            arr = np.load(npy_path)          # (K, 4)  [cx cy cz r]
            self.sphere_tensors.append(
                torch.tensor(arr, dtype=self.dtype, device=device)
            )

        # -- KD-tree: built once, never rebuilt ----------------------------
        if torch.is_tensor(obstacle_points):
            self._obs_pts = obstacle_points.to(device, dtype=self.dtype)
        else:
            self._obs_pts = torch.tensor(
                obstacle_points, dtype=self.dtype, device=device
            )
        self._kdtree = torch_kdtree.build_kd_tree(self._obs_pts)
    def _sphere_obs_dist(self, q, link_idx, sphere_idx, obs_idx):
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device).unsqueeze(0)
        tg = self.chain.forward_kinematics(q_t, end_only=False)
        M = tg[_LINK_NAMES[link_idx]].get_matrix()[0]

        sphere = self.sphere_tensors[link_idx][sphere_idx]
        cxyz = sphere[:3]
        radius = sphere[3]

        pt_h = torch.cat([
            cxyz,
            torch.ones(1, device=self.device, dtype=self.dtype),
        ])

        world_pt = (M @ pt_h)[:3]
        obs_pt = self._obs_pts[obs_idx]

        return (torch.norm(world_pt - obs_pt) - radius).item()
    def query(self, q):
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device).unsqueeze(0)
        tg_batch = self.chain.forward_kinematics(q_t, end_only=False)
        matrices = [tg_batch[name].get_matrix() for name in _LINK_NAMES]

        N_VALID = N_NEIGHBORS
        n_constraints = N_LINKS * N_VALID

        mu_list = np.zeros(n_constraints)
        grad_list = np.zeros((n_constraints, 6))
        obs_k_list = np.zeros(n_constraints, dtype=int)

        eps = 1e-4

        for i, spheres in enumerate(self.sphere_tensors):
            M = matrices[i][0]
            cxyz = spheres[:, :3]
            radii = spheres[:, 3]

            ones = torch.ones(
                cxyz.shape[0], 1,
                device=self.device,
                dtype=self.dtype,
            )
            pts_h = torch.cat([cxyz, ones], dim=1)
            world_pts = (M @ pts_h.T).T[:, :3]

            dists_sq, inds = self._kdtree.query(
                world_pts,
                nr_nns_searches=N_NEIGHBORS,
            )

            signed_d = torch.sqrt(dists_sq) - radii[:, None]

            # Pick the N_VALID smallest sphere-obstacle distances for this link.
            flat_vals = signed_d.reshape(-1)
            top_vals, top_flat_idx = torch.topk(
                flat_vals,
                k=N_VALID,
                largest=False,
            )

            num_neighbors = signed_d.shape[1]

            for r in range(N_VALID):
                idx = i * N_VALID + r

                flat_idx = int(top_flat_idx[r].item())
                sphere_idx = flat_idx // num_neighbors
                neighbor_idx = flat_idx % num_neighbors

                mu_val = float(top_vals[r].item())
                obs_idx = int(inds[sphere_idx, neighbor_idx].item())

                mu_list[idx] = mu_val
                obs_k_list[idx] = obs_idx

                if mu_val < GRAD_THRESHOLD:
                    grad_row = np.zeros(6, dtype=np.float64)

                    for j in range(6):
                        dq = np.zeros(6, dtype=np.float64)
                        dq[j] = eps

                        grad_row[j] = (
                            self._sphere_obs_dist(q + dq, i, sphere_idx, obs_idx)
                            - self._sphere_obs_dist(q - dq, i, sphere_idx, obs_idx)
                        ) / (2.0 * eps)

                    grad_list[idx] = grad_row

        sigma_list = np.full(n_constraints, SIGMA_CONST, dtype=np.float64)

        return mu_list, sigma_list, grad_list, obs_k_list
    def query_old(self, q):
        """
        Parameters
        ----------
        q : (6,) joint configuration [rad]  numpy float64

        Returns
        -------
        mu    : (N_LINKS,)
        sigma : (N_LINKS,)   all == SIGMA_CONST
        grad  : (N_LINKS, 6)
        obs_k : (N_LINKS,)   closest obstacle point index per link
        """
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device).unsqueeze(0)

        # FK for all links at once
        tg_batch = self.chain.forward_kinematics(q_t, end_only=False)
        matrices = [tg_batch[name].get_matrix() for name in _LINK_NAMES]

        eps        = 1e-4
        mu_list    = np.zeros(N_LINKS)
        grad_list  = np.zeros((N_LINKS, 6))
        obs_k_list = np.zeros(N_LINKS, dtype=int)

        for i, spheres in enumerate(self.sphere_tensors):
            M     = matrices[i][0]           # (4,4)
            cxyz  = spheres[:, :3]           # (K,3)
            radii = spheres[:, 3]            # (K,)

            ones      = torch.ones(cxyz.shape[0], 1, device=self.device, dtype=self.dtype)
            pts_h     = torch.cat([cxyz, ones], dim=1)         # (K,4)
            world_pts = (M @ pts_h.T).T[:, :3]                 # (K,3)

            dists_sq, inds = self._kdtree.query(world_pts, nr_nns_searches=16)
            print(dists_sq.shape, inds.shape)
            signed_d = torch.sqrt(dists_sq.squeeze(1)) - radii # (K,)

            min_idx       = torch.argmin(signed_d).item()
            mu_list[i]    = signed_d[min_idx].item()
            obs_k_list[i] = inds[min_idx].item()

            # Central-difference gradient, pinning the closest obstacle point
            obs_idx  = obs_k_list[i]
            grad_row = np.zeros(6)
            for j in range(6):
                dq    = np.zeros(6, dtype=np.float64)
                dq[j] = eps
                grad_row[j] = (
                    self._link_min_dist(q + dq, i, obs_idx)
                    - self._link_min_dist(q - dq, i, obs_idx)
                ) / (2.0 * eps)
            grad_list[i] = grad_row

        return mu_list, np.full(N_LINKS, SIGMA_CONST), grad_list, obs_k_list

    def _link_min_dist(self, q, link_idx, obs_idx):
        """Distance from link's nearest sphere to one fixed obstacle point."""
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device).unsqueeze(0)
        tg  = self.chain.forward_kinematics(q_t, end_only=False)
        M   = tg[_LINK_NAMES[link_idx]].get_matrix()[0]

        spheres   = self.sphere_tensors[link_idx]
        cxyz      = spheres[:, :3]
        radii     = spheres[:, 3]
        ones      = torch.ones(cxyz.shape[0], 1, device=self.device, dtype=self.dtype)
        pts_h     = torch.cat([cxyz, ones], dim=1)
        world_pts = (M @ pts_h.T).T[:, :3]

        obs_pt = self._obs_pts[obs_idx]
        dists  = torch.norm(world_pts - obs_pt.unsqueeze(0), dim=1) - radii
        return torch.min(dists).item()


# -- Public API ---------------------------------------------------------------

def mu_sigma_grad_mesh(q, model):
    """
    Drop-in replacement for mu_sigma_grad_nn.
    obstacle_points no longer needed here - already inside model.

    Parameters
    ----------
    q     : (6,) current joint config [rad]  - numpy or torch
    model : MeshDistanceModel

    Returns
    -------
    mu, sigma, grad, obs_k
    """
    if torch.is_tensor(q):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float64).reshape(6)
    return model.query(q)