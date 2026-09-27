"""
mesh_inference.py

Sphere-approximated robot geometry + KD-tree over the obstacle point cloud,
with autograd-based joint-space gradients.

Shape contract (matches mu_sigma_grad_nn):
    mu    : (N_LINKS * N_VALID,)        signed sphere-to-obstacle distances [m]
    sigma : (N_LINKS * N_VALID,)        constant SIGMA_CONST
    grad  : (N_LINKS * N_VALID, 6)      d(distance)/dq  in joint space
    obs_k : (N_LINKS * N_VALID,)        index of the closest obstacle point per constraint

Usage
-----
    from mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh
    mesh_model = MeshDistanceModel(
        urdf_path       = "datasets/arm/UR5/ur5e.urdf",
        sphere_dir      = "datasets/arm/UR5/meshes/sphere/sphere",
        obstacle_points = obs_array,       # (M, 3) - fixed for the session
        end_link        = "wrist_3_link",
        device          = "cuda",
    )

    mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q0, mesh_model)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch_kdtree

# CHANGED: pytorch_kinematics dropped in favor of our hand-rolled FK.
from mesh_inference.serial_chain_fk import SerialChainFK


# ── Module-level constants ──────────────────────────────────────────────────
SIGMA_CONST    = 0.028 #0.03     # fixed noise level [m]
N_NEIGHBORS    = 10       # torch_kdtree may round up internally
GRAD_THRESHOLD = 0.10     # zero out gradients beyond this distance [m]

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
    All heavy structures (FK, sphere approximations, KD-tree) are built once
    at construction. The obstacle cloud is assumed fixed for the lifetime of
    this object.
    """

    def __init__(
        self,
        urdf_path,
        sphere_dir,
        obstacle_points,
        end_link="wrist_3_link",          # kept for API compatibility; unused
        device="cuda",
        camera_origin=(0.2091, 0.2957, 0.2582),   # camera frame origin with respect to robot base
        sigma_coeff=0.06,                           # camera noise model: sigma = coeff * d^2
    ):
        self.device = device
        self.dtype  = torch.float32

        # -- Forward kinematics (differentiable) ---------------------------
        # CHANGED: ~400 ops/call (pytorch_kinematics) -> ~12 ops/call (ours).
        # See serial_chain_fk.py for parsing + flat torch chain.
        self.fk = SerialChainFK(
            urdf_path  = urdf_path,
            link_names = _LINK_NAMES,
            device     = device,
            dtype      = self.dtype,
        )

        # -- Sphere approximations -----------------------------------------
        # CHANGED: precompute homogeneous coords and radii so query() never
        # has to torch.cat / torch.ones on the hot path.
        self.sphere_tensors = []          # kept for backward compatibility
        self._sphere_h      = []          # (K_sph_i, 4)  homogeneous centers
        self._sphere_radii  = []          # (K_sph_i,)    sphere radii
        for name in _LINK_NAMES:
            npy_path = os.path.join(sphere_dir, f"{name}.npy")
            arr = np.load(npy_path)                      # (K, 4): [cx cy cz r]
            t = torch.tensor(arr, dtype=self.dtype, device=device)
            self.sphere_tensors.append(t)

            cxyz = t[:, :3]
            ones = torch.ones(cxyz.shape[0], 1, dtype=self.dtype, device=device)
            self._sphere_h.append(torch.cat([cxyz, ones], dim=1))
            self._sphere_radii.append(t[:, 3].contiguous())

        # -- KD-tree: built once, never rebuilt ----------------------------
        if torch.is_tensor(obstacle_points):
            self._obs_pts = obstacle_points.to(device, dtype=self.dtype)
        else:
            self._obs_pts = torch.tensor(
                obstacle_points, dtype=self.dtype, device=device
            )
        self._kdtree = torch_kdtree.build_kd_tree(self._obs_pts)

    def query(self, q):
        """
        Single forward pass + one vmap-batched backward pass.

        Returns
        -------
        mu    : (N_LINKS * N_VALID,)       float64
        sigma : (N_LINKS * N_VALID,)       float64, constant SIGMA_CONST
        grad  : (N_LINKS * N_VALID, 6)     float64
        obs_k : (N_LINKS * N_VALID,)       int, obstacle index per constraint
        """
        # Track gradients through joint configuration.
        q_t = torch.tensor(
            q, dtype=self.dtype, device=self.device,
        ).unsqueeze(0).requires_grad_(True)

        # CHANGED: one-line FK. Returns (1, N_LINKS, 4, 4) world-frame xforms.
        T_world = self.fk(q_t)

        N_VALID       = N_NEIGHBORS
        n_constraints = N_LINKS * N_VALID

        all_d, all_obs_idx = [], []

        for i in range(N_LINKS):
            M     = T_world[0, i]                                    # (4, 4)
            pts_h = self._sphere_h[i]                                # (K_sph, 4)  precomputed
            radii = self._sphere_radii[i]                            # (K_sph,)    precomputed

            # (pts_h @ M.T) is equivalent to (M @ pts_h.T).T,
            # one fewer transpose op in the graph.
            world_pts = (pts_h @ M.T)[:, :3]                         # (K_sph, 3)

            # KD-tree is non-differentiable; use it only to pick indices.
            with torch.no_grad():
                _, inds = self._kdtree.query(
                    world_pts.detach(),
                    nr_nns_searches=N_NEIGHBORS,
                )                                                    # (K_sph, N_NN)

            # Recompute distances in a differentiable form.
            obs_pts_sel = self._obs_pts[inds]                        # (K_sph, N_NN, 3)
            diffs       = world_pts.unsqueeze(1) - obs_pts_sel       # (K_sph, N_NN, 3)
            # +1e-12 inside the sqrt keeps the gradient well-defined when
            # a sphere center coincides with an obstacle point.
            norms       = torch.sqrt((diffs * diffs).sum(dim=-1) + 1e-12)
            signed_d    = norms - radii[:, None]                     # (K_sph, N_NN)

            flat_d    = signed_d.reshape(-1)
            flat_inds = inds.reshape(-1)

            top_vals, top_flat_idx = torch.topk(
                flat_d, k=N_VALID, largest=False,
            )

            all_d.append(top_vals)
            all_obs_idx.append(flat_inds[top_flat_idx])

        all_d_t       = torch.cat(all_d)                             # (N_constraints,)
        all_obs_idx_t = torch.cat(all_obs_idx)                       # (N_constraints,)

        # ── One vmap'd backward for all K gradients ───────────────────────
        K_total = all_d_t.numel()
        near_mask = all_d_t < GRAD_THRESHOLD
        near_idx  = near_mask.nonzero(as_tuple=True)[0]

        grads = torch.zeros((K_total, 6), device=self.device, dtype=self.dtype)
        if near_idx.numel() > 0:
            near_d = all_d_t[near_idx]
            I_near = torch.eye(near_d.numel(), device=self.device, dtype=self.dtype)
            near_g = torch.autograd.grad(
                near_d, q_t,
                grad_outputs=I_near,
                is_grads_batched=True,
            )[0].squeeze(1)
            grads[near_idx] = near_g

        mu    = all_d_t.detach().cpu().numpy().astype(np.float64)
        grad  = grads.detach().cpu().numpy().astype(np.float64)
        obs_k = all_obs_idx_t.detach().cpu().numpy().astype(int)
        sigma = np.full(n_constraints, SIGMA_CONST, dtype=np.float64)

        return mu, sigma, grad, obs_k


# ── Public API ──────────────────────────────────────────────────────────────

def mu_sigma_grad_mesh(q, model):
    """
    Drop-in replacement for mu_sigma_grad_nn.
    obstacle_points are already inside `model`.

    Parameters
    ----------
    q     : (6,) joint configuration [rad]  -  numpy or torch
    model : MeshDistanceModel

    Returns
    -------
    mu, sigma, grad, obs_k
    """
    if torch.is_tensor(q):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float64).reshape(6)
    return model.query(q)