import os
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import math
import matplotlib.pyplot as plt

import torch_kdtree  # import build_kd_tree

from torch_IK_UR5 import torch_IK_UR5, transformRobotParameter

from functorch import vmap, jacfwd

# ---------------------------------------------------------------------------- #
#  Risk-sensitive distance configuration                                       #
# ---------------------------------------------------------------------------- #
#
#  TWO independent confidence levels control the two stages of the formulation:
#
#  ALPHA_VAR ─ controls the per-pair Gaussian VaR shift:
#       d_VaR^(i)(q, x) = μ_i(q, x) + σ_i(q, x) · Φ^{-1}(ALPHA_VAR)
#     • ALPHA_VAR = 0.5   →  Φ^{-1} = 0          →  VaR = μ (no σ shift at all)
#     • ALPHA_VAR = 0.05  →  Φ^{-1} ≈ -1.6449    →  5% lower-tail (paper default)
#
#  ALPHA_CVAR ─ controls the per-configuration CVaR aggregation:
#       d_risk(q) = mean { VaR pairs ≤ τ_{ALPHA_CVAR}(q) }
#     • ALPHA_CVAR = 0.0  →  τ = min over pairs  →  CVaR ≡ hard min (argmin pair)
#     • ALPHA_CVAR = 0.05 →  mean of the worst 5% of pairs (paper default)
#
#  To RECOVER THE ORIGINAL (deterministic, hard-min) BEHAVIOR exactly, set:
#       ALPHA_VAR  = 0.5
#       ALPHA_CVAR = 0.0
#  Under those values, d_risk(q) ≡ min_i μ_i(q) and ∇d_risk ≡ normal of argmin.
#
#  σ comes from a depth-dependent camera noise model, evaluated at the nearest
#  obstacle point selected by the KD-tree for each link sphere:
#       σ(x) = SIGMA_BIAS + SIGMA_COEFF · ||x - camera||^2
# ---------------------------------------------------------------------------- #
ALPHA_VAR   = 0.5                                    # per-pair VaR confidence
ALPHA_CVAR  = 0.0                                    # per-config CVaR confidence
SIGMA_BIAS  = 0.002                                   # σ floor (meters)
SIGMA_COEFF = 0.0015                                  # quadratic depth coefficient
CAMERA_POS  = torch.tensor([0.0, 0.0, 0.5],           # depth camera position
                           dtype=torch.float32, device='cuda')
# Φ^{-1}(ALPHA_VAR) precomputed once at module load:
#   Φ^{-1}(α) = sqrt(2) · erfinv(2α - 1)
PHI_INV_ALPHA_VAR = float(math.sqrt(2.0)
                          * torch.erfinv(torch.tensor(2.0 * ALPHA_VAR - 1.0)))


def depth_dependent_sigma(points, camera_pos=CAMERA_POS):
    """σ(x) = SIGMA_BIAS + SIGMA_COEFF · ||x - camera||^2.

    Args:
        points: (..., 3) tensor of 3D points (on the obstacle surface).
        camera_pos: (3,) camera position.
    Returns:
        (...,) tensor of per-point standard deviations.
    """
    d_cam = torch.norm(points - camera_pos, dim=-1)
    return SIGMA_BIAS + SIGMA_COEFF * d_cam.pow(2)


def viz_two_poses(XP, env_path="datasets/arm/UR5/realpc_scaled.off"):
    """
    Visualize the two UR5 poses packed into XP (shape 1×12, raw radians).
    Uses the same FK / chain / mesh_list as the rest of this file.
    """
    import numpy as np

    scale = np.pi / 0.5                          # same scale used everywhere

    # Split into two (1, 6) configs — raw radians
    cfg_a = XP[:, :6]                            # first pose
    cfg_b = XP[:, 6:]                            # second pose
    joint_configs = torch.cat([cfg_a, cfg_b], dim=0)   # (2, 6)

    # Normalize so viz_ik_solutions_with_arm can re-scale internally
    joint_configs_norm = joint_configs / scale

    chain, mesh_list = build_chain()

    # Show EE Cartesian positions (FK of both configs)
    viz_sampling_debug(
        env_path=env_path,
        ik_points_fk=joint_configs_norm,
        chain=chain,
        scale=scale,
    )

    # Show full arm geometry for both configs (n=2 → one color each)
    viz_ik_solutions_with_arm(
        env_path=env_path,
        joint_configs=joint_configs_norm,
        chain=chain,
        mesh_list=mesh_list,
        scale=scale,
        n=2,
    )


def viz_sampling_debug(env_path, ee_points=None, ik_points_fk=None, chain=None, scale=None):
    """
    Visualize obstacle point cloud + end-effector samples + FK positions of IK solutions.

    Args:
        env_path:      path to the obstacle .off mesh
        ee_points:     (N,3) tensor - raw end-effector Cartesian samples (P before IK)
        ik_points_fk:  (N,3) tensor - FK of IK solutions (to verify IK correctness)
        chain:         pytorch_kinematics chain (needed to compute FK for ik_points_fk)
        scale:         scale factor (pi/0.5) to undo normalization before FK
    """
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    import igl

    v, f = igl.read_triangle_mesh(env_path)
    print('obstacle bbox:', v.min(axis=0), v.max(axis=0))

    z_values = v[:, 2]
    x_values = v[:, 0]
    combined = (z_values + x_values) / 2
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min())
    cmap = plt.get_cmap('viridis')
    pc_colors = (cmap(combined_norm) * 255).astype(np.uint8)

    point_cloud = trimesh.points.PointCloud(vertices=v, colors=pc_colors)

    scene = trimesh.Scene()
    scene.add_geometry(point_cloud)

    axis_length = 0.5
    for vec, col in [([axis_length, 0, 0], [255, 0, 0, 255]),
                     ([0, axis_length, 0], [0, 255, 0, 255]),
                     ([0, 0, axis_length], [0, 0, 255, 255])]:
        ax = trimesh.load_path(np.array([[0, 0, 0], vec]))
        ax.colors = [col]
        scene.add_geometry(ax)

    if ee_points is not None:
        pts = ee_points.detach().cpu().numpy() if hasattr(ee_points, 'detach') else ee_points
        print(f'Visualizing {len(pts)} end-effector samples')
        spheres = []
        for pt in pts[::max(1, len(pts) // 500)]:
            s = trimesh.creation.icosphere(radius=0.015)
            s.apply_translation(pt)
            s.visual.vertex_colors = np.tile([255, 200, 0, 180], (len(s.vertices), 1))
            spheres.append(s)
        scene.add_geometry(trimesh.util.concatenate(spheres))

    if ik_points_fk is not None and chain is not None and scale is not None:
        joints = ik_points_fk.detach().cpu() if hasattr(ik_points_fk, 'detach') else torch.tensor(ik_points_fk)
        joints_scaled = (joints * scale).to('cuda')

        with torch.no_grad():
            tg = chain.forward_kinematics(joints_scaled, end_only=True)
            fk_positions = tg.get_matrix()[:, :3, 3].cpu().numpy()

        print(f'Visualizing {len(fk_positions)} FK-verified IK positions')
        spheres = []
        for pt in fk_positions[::max(1, len(fk_positions) // 500)]:
            s = trimesh.creation.icosphere(radius=0.015)
            s.apply_translation(pt)
            s.visual.vertex_colors = np.tile([0, 200, 255, 180], (len(s.vertices), 1))
            spheres.append(s)
        scene.add_geometry(trimesh.util.concatenate(spheres))

    scene.show()


def viz_ik_solutions_with_arm(env_path, joint_configs, chain, mesh_list, scale, n=10):
    """
    Randomly select n IK solutions and visualize:
      - Obstacle point cloud
      - End-effector position as a cyan sphere
      - Full arm geometry as colored bounding spheres (one color per sampled config)
    """
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    import igl

    v, f = igl.read_triangle_mesh(env_path)
    z_values = v[:, 2]
    x_values = v[:, 0]
    combined = (z_values + x_values) / 2
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min())
    cmap = plt.get_cmap('viridis')
    pc_colors = (cmap(combined_norm) * 255).astype(np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(vertices=v, colors=pc_colors))

    axis_length = 0.5
    for vec, col in [([axis_length, 0, 0], [255, 0, 0, 255]),
                     ([0, axis_length, 0], [0, 255, 0, 255]),
                     ([0, 0, axis_length], [0, 0, 255, 255])]:
        ax = trimesh.load_path(np.array([[0, 0, 0], vec]))
        ax.colors = [col]
        scene.add_geometry(ax)

    configs = joint_configs.detach().cpu() if hasattr(joint_configs, 'detach') else torch.tensor(joint_configs)
    N = configs.shape[0]
    n = min(n, N)
    indices = torch.randperm(N)[:n]
    selected = configs[indices]

    color_cmap = plt.get_cmap('tab10')
    config_colors = [(np.array(color_cmap(i % 10)) * 255).astype(np.uint8) for i in range(n)]

    joints_scaled = (selected * scale).to('cuda')

    with torch.no_grad():
        tg_batch = chain.forward_kinematics(joints_scaled, end_only=False)
        tg_ee = chain.forward_kinematics(joints_scaled, end_only=True)
        ee_positions = tg_ee.get_matrix()[:, :3, 3].cpu().numpy()

    all_spheres = []

    for i in range(n):
        color = config_colors[i]
        arm_color = np.array([color[0], color[1], color[2], 160], dtype=np.uint8)
        ee_color = np.array([0, 220, 255, 220], dtype=np.uint8)

        link_idx = 0
        for link_name in tg_batch:
            if link_idx > 1 and link_idx < 8:
                ball_list = mesh_list[link_idx - 2]
                m = tg_batch[link_name].get_matrix()

                ones = torch.ones(ball_list.shape[0], 1, device='cuda')
                nv = torch.cat([ball_list[:, :3], ones], dim=1)

                m_i = m[i]
                p = (m_i @ nv.T).T[:, :3].cpu().numpy()
                r = ball_list[:, 3].cpu().numpy()

                for center, radius in zip(p, r):
                    s = trimesh.creation.icosphere(radius=float(radius), subdivisions=2)
                    s.apply_translation(center)
                    s.visual.vertex_colors = np.tile(arm_color, (len(s.vertices), 1))
                    all_spheres.append(s)

            link_idx += 1

        s = trimesh.creation.icosphere(radius=0.02, subdivisions=2)
        s.apply_translation(ee_positions[i])
        s.visual.vertex_colors = np.tile(ee_color, (len(s.vertices), 1))
        all_spheres.append(s)

    if all_spheres:
        scene.add_geometry(trimesh.util.concatenate(all_spheres))

    print(f'Visualizing {n} IK solutions with full arm geometry')
    scene.show()


def rot_pose(x, a, b, c):

    rot_pose = torch.zeros((x.shape[0], 4, 4), dtype=torch.float32, device='cuda')
    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    cos_b = torch.cos(b)
    sin_b = torch.sin(b)
    cos_c = torch.cos(c)
    sin_c = torch.sin(c)

    rot_pose[:, 0, 0] = cos_b * cos_c
    rot_pose[:, 0, 1] = sin_a * sin_b * cos_c - cos_a * sin_c
    rot_pose[:, 0, 2] = cos_a * sin_b * cos_c + sin_a * sin_c

    rot_pose[:, 1, 0] = cos_b * sin_c
    rot_pose[:, 1, 1] = sin_a * sin_b * sin_c + cos_a * cos_c
    rot_pose[:, 1, 2] = cos_a * sin_b * sin_c - sin_a * cos_c

    rot_pose[:, 2, 0] = -sin_b
    rot_pose[:, 2, 1] = sin_a * cos_b
    rot_pose[:, 2, 2] = cos_a * cos_b

    rot_pose[:, :3, 3] = x

    rot_pose[:, 3, 3] = 1
    return rot_pose


def build_chain():
    out_path = 'datasets/arm/UR5'
    end_effect = 'wrist_3_link'
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    chain = pk.build_serial_chain_from_urdf(
        open(out_path + '/' + "ur5e.urdf").read(), end_effect)
    chain = chain.to(dtype=dtype, device=d)
    th_batch = torch.rand(1, 6, device='cuda')
    tg_batch = chain.forward_kinematics(th_batch, end_only=False)
    mesh_list = []
    iter = 0
    for tg in tg_batch:
        if iter > 1 and iter < 8:
            ball_list = np.load(out_path + '/meshes/sphere/sphere/' + tg + '.npy')
            mesh_list.append(torch.tensor(ball_list, dtype=torch.float32, device='cuda'))
        iter = iter + 1
    return chain, mesh_list


def gradient(y, x, create_graph=True):

    grad_y = torch.ones_like(y)

    grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]

    return grad_x


def FK(sampled_points, chain, mesh_list):
    shape = sampled_points.shape
    pointsize = 0

    scale = np.pi / 0.5

    th_batch = torch.tensor(sampled_points, requires_grad=True, device='cuda')
    del sampled_points

    def chain_to_matrix(th_batch):

        tg_batch = chain.forward_kinematics(th_batch, end_only=False)
        iter = 0
        m_list = []
        for tg in tg_batch:
            if iter > 1 and iter < 8:
                m = tg_batch[tg].get_matrix()
                m_list.append(m)
                del m
            iter = iter + 1
        del tg_batch, th_batch
        return torch.cat(m_list, dim=0)

    def output_and_jacobian_fn(th_batch):
        output = chain_to_matrix(th_batch)
        jacobian = jacfwd(chain_to_matrix)(th_batch)
        del th_batch
        return output, jacobian

    matrix_list, jacobian = vmap(output_and_jacobian_fn)(th_batch)
    matrix_list = matrix_list.detach()
    jacobian = jacobian.detach()
    del th_batch

    torch.cuda.empty_cache()
    iter = 0
    p_list = []
    gradient_p_list = []
    for iter in range(6):

        ball_list = mesh_list[iter]

        ones_column = torch.ones(ball_list.size(0), 1, device='cuda')
        nv = torch.cat((ball_list[:, :3], ones_column), dim=1)

        m = matrix_list[:, iter, ...]
        gradient_m = jacobian[:, iter, ...].permute(0, 3, 1, 2)

        print(m.shape, nv.shape)

        p = torch.matmul(m[:], nv.T)
        p = torch.permute(p, (0, 2, 1))
        p[..., 3] = ball_list[:, 3]
        p_list.append(p)

        gradient_p = torch.matmul(gradient_m, nv.T)
        gradient_p_list.append(gradient_p)

        del m, p, nv, ones_column, ball_list, gradient_p, gradient_m
    del matrix_list, jacobian
    torch.cuda.empty_cache()
    return p_list, gradient_p_list


def arm_obstacle_distance(th_batch, chain, mesh_list, kdtree, v_obs,
                          alpha_cvar=ALPHA_CVAR,
                          camera_pos=CAMERA_POS,
                          phi_inv_alpha_var=PHI_INV_ALPHA_VAR):
    """
    Risk-sensitive arm-to-obstacle clearance.

    For each joint configuration q in th_batch:
      1. Forward-kinematics the bounding spheres of links 2..7.
      2. For each sphere, find the nearest obstacle point via KD-tree.
         μ_i(q, x) = sqrt(d²) - radius                    (geometric clearance)
         σ_i(q, x) = SIGMA_BIAS + SIGMA_COEFF·||x - cam||²
         d_VaR^(i)(q, x) = μ_i + σ_i · Φ^{-1}(ALPHA_VAR)  (Gaussian closed form)
      3. CVaR-aggregate the per-sphere VaR values at level ALPHA_CVAR:
         d_risk(q) = mean of VaR values below the α-quantile τ_α(q).
      4. The gradient w.r.t. q is the average of per-pair gradients restricted to
         the same tail set (σ does not depend on q, so ∇VaR = ∇μ at the
         locally-fixed nearest point).

    Setting ALPHA_VAR = 0.5 and ALPHA_CVAR = 0.0 recovers the original
    deterministic hard-min behavior exactly.

    Returns:
        whole_dis:    (B,)    d_risk per configuration.
        whole_normal: (B, 6)  ∇d_risk w.r.t. joint angles per configuration.
    """
    whole_dis = torch.zeros((th_batch.shape[0]), dtype=torch.float32, device='cuda')
    whole_normal = torch.zeros((th_batch.shape[0], 6), dtype=torch.float32, device='cuda')
    end_dis = []

    batch_size = 200000

    for batch_id in range(math.floor(th_batch.shape[0] / batch_size) + 1):
        if batch_id * batch_size == th_batch.shape[0]:
            break

        sampled_points = th_batch[batch_id * batch_size:
                                  min((batch_id + 1) * batch_size, th_batch.shape[0]), :]
        batch_size = sampled_points.shape[0]

        p_list, gradient_p_list = FK(sampled_points, chain, mesh_list)
        del sampled_points

        torch.cuda.empty_cache()

        query_points = torch.cat(p_list, dim=1)
        query_points_grad = torch.cat(gradient_p_list, dim=3)
        del p_list, gradient_p_list

        torch.cuda.empty_cache()

        query_points_grad = query_points_grad.permute(0, 3, 1, 2)

        query_points = torch.reshape(query_points, (-1, 4))
        query_points_grad = torch.reshape(query_points_grad, (-1, 6, 4))

        # ----- nearest obstacle point per sphere ---------------------------- #
        dists, inds = kdtree.query(query_points[:, :3], nr_nns_searches=1)
        dists = dists.squeeze()
        inds = inds.squeeze()

        # μ_i(q, x): geometric clearance per (sphere, nearest-obstacle) pair
        mu = torch.sqrt(dists) - query_points[:, 3]

        del dists

        # ----- σ_i(q, x): depth-dependent obstacle noise -------------------- #
        nearest_obs = v_obs[inds, :]                          # (N_pairs, 3)
        sigma = depth_dependent_sigma(nearest_obs, camera_pos)  # (N_pairs,)

        # ----- per-pair VaR under Gaussian assumption ----------------------- #
        # d_VaR^(i)(q, x) = μ + σ · Φ^{-1}(ALPHA_VAR)
        var_per_pair = mu + sigma * phi_inv_alpha_var

        # ----- per-pair gradient ∇VaR = ∇μ (σ independent of q) ------------- #
        # Vector from nearest obstacle point to sphere center, then chain-ruled
        # through the link Jacobian to produce a 6-DoF gradient.
        normal = query_points[:, :3] - nearest_obs
        normal = torch.bmm(query_points_grad[..., :3], normal.unsqueeze(2)).squeeze(2)

        del inds, query_points, nearest_obs, sigma, mu

        torch.cuda.empty_cache()

        # Reshape to (B, total_spheres) and (B, total_spheres, 6)
        var_per_pair = var_per_pair.reshape(batch_size, -1)
        normal = normal.reshape(batch_size, -1, normal.shape[-1])

        # ----- CVaR aggregation across pairs per configuration -------------- #
        # τ_α(q): empirical ALPHA_CVAR-quantile of per-pair VaR values per config.
        tau_alpha = torch.quantile(var_per_pair, alpha_cvar, dim=1, keepdim=True)  # (B, 1)
        # Tail set: pairs whose VaR is at or below τ_α.
        mask = (var_per_pair <= tau_alpha).to(var_per_pair.dtype)             # (B, total_spheres)
        denom = mask.sum(dim=1).clamp(min=1.0)                                # avoid div-by-zero

        # d_risk(q) = mean of VaR values in the tail set
        cvar_distance = (var_per_pair * mask).sum(dim=1) / denom              # (B,)

        # ∇d_risk(q) = mean of per-pair ∇VaR over the tail set
        mask_exp = mask.unsqueeze(-1)                                          # (B, total_spheres, 1)
        cvar_normal = (normal * mask_exp).sum(dim=1) / denom.unsqueeze(-1)     # (B, 6)

        del var_per_pair, normal, mask, mask_exp, tau_alpha, denom

        torch.cuda.empty_cache()

        whole_dis[batch_id * batch_size: min((batch_id + 1) * batch_size, th_batch.shape[0])] = cvar_distance
        whole_normal[batch_id * batch_size: min((batch_id + 1) * batch_size, th_batch.shape[0]), :] = cvar_normal

        del cvar_distance, cvar_normal

        torch.cuda.empty_cache()

    del th_batch

    torch.cuda.empty_cache()

    return whole_dis, whole_normal


def arm_append_list(X_list, Y_list, N_list,
                    chain, mesh_list, kdtree, v_obs,
                    numsamples, dim, offset, margin):

    OutsideSize = numsamples + 2
    WholeSize = 0

    scale = math.pi / 0.5

    while OutsideSize > 0:

        P = torch.rand((int(2 * numsamples), 3), dtype=torch.float32, device='cuda')
        P[:, 0] = (P[:, 0] * 1.2 + 0.2)
        P[:, 1] = (P[:, 1] - 0.5) * 1.2
        P[:, 2] = P[:, 2] * 1.2 - 0.1
        XP = torch.tensor([[0.3, -0.7, -1.0, 0.5 * np.pi, 0.5 * np.pi, 0.0,
                            -0.2, -0.5, -0.35, 0.2 * np.pi, 0.5 * np.pi, 0.0]]).cuda()
        BASE = torch.tensor([[0, -0.5 * np.pi, 0.0, -0.5 * np.pi, 0.0, 0.0,
                              0, -0.5 * np.pi, 0.0, -0.5 * np.pi, 0.0, 0.0]]).cuda()
        XP = XP + BASE
        viz_two_poses(XP)
        x0 = P

        a0 = -0.5 * math.pi + (torch.rand((x0.shape[0], 1), dtype=torch.float32, device='cuda') - 0.5) * 0.6 * math.pi
        b0 = (torch.rand((x0.shape[0], 1), dtype=torch.float32, device='cuda') - 0.5) * 0.6 * math.pi
        c0 = -0.5 * math.pi + (torch.rand((x0.shape[0], 1), dtype=torch.float32, device='cuda') - 0.5) * 0.6 * math.pi

        end_pose0 = rot_pose(x0, a0.squeeze(), b0.squeeze(), c0.squeeze())
        print(end_pose0.shape)
        torch_ik = torch_IK_UR5(end_pose0.shape[0])
        torch_ik.setJointLimits(-math.pi, math.pi)
        t0 = torch_ik.solveIK(end_pose0)
        x0 = torch.reshape(t0, (t0.shape[0] * t0.shape[1], t0.shape[2]))
        del end_pose0, t0, torch_ik, a0, b0, c0

        x0 = x0 / scale
        viz_sampling_debug(
            env_path="datasets/arm/UR5/realpc_scaled.off",
            ee_points=P,
            ik_points_fk=x0,
            chain=chain,
            scale=scale,
        )

        dP = torch.rand((x0.shape[0], dim), dtype=torch.float32, device='cuda') - 0.5
        rL = (torch.rand((x0.shape[0], 1), dtype=torch.float32, device='cuda')) * 0.5
        nP = x0 + torch.nn.functional.normalize(dP, dim=1) * rL
        del P, dP, rL
        PointsInside = torch.all((nP <= 0.5), dim=1) & torch.all((nP >= -0.5), dim=1)

        x0 = x0[PointsInside, :]
        x1 = nP[PointsInside, :]
        print('After IK + PointsInside filter:', x0.shape)
        if(x0.shape[0] <= 1):
            continue

        th_batch0 = scale * x0
        # Risk-sensitive (CVaR of per-pair VaR) clearance per config
        obs_distance0, normal0 = arm_obstacle_distance(th_batch0, chain, mesh_list, kdtree, v_obs)
        obs_distance0 = obs_distance0 -0.018

        where_d = (obs_distance0 > 0) & (obs_distance0 < margin)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        n0 = normal0[where_d]

        th_batch1 = scale * x1
        obs_distance1, normal1 = arm_obstacle_distance(th_batch1, chain, mesh_list, kdtree, v_obs)
        obs_distance1 = obs_distance1 -0.018

        y1 = obs_distance1
        n1 = normal1

        del th_batch0, th_batch1, obs_distance0, obs_distance1, normal0, normal1

        print('WholeSize', WholeSize)
        n0 = n0 / torch.norm(n0, dim=1, keepdim=True)
        n1 = n1 / torch.norm(n1, dim=1, keepdim=True)
        where0 = ~torch.any(n0.isnan(), dim=1)
        where1 = ~torch.any(n1.isnan(), dim=1)
        x0 = x0[where0 & where1]
        x1 = x1[where0 & where1]
        y0 = y0[where0 & where1]
        y1 = y1[where0 & where1]
        n0 = n0[where0 & where1]
        n1 = n1[where0 & where1]

        x = torch.cat((x0, x1), 1)
        y = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1)), 1)
        n = torch.cat((n0, n1), 1)
        del x0, x1, y0, y1, n0, n1

        X_list.append(x)
        Y_list.append(y)
        N_list.append(n)

        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]

        del x, y, n

        if(WholeSize > numsamples):
            break
    return X_list, Y_list, N_list


def arm_rand_sample_bound_points(numsamples, dim,
                                 v_obs, offset, margin):
    numsamples = int(numsamples)

    chain, mesh_list = build_chain()

    bb_max = np.array([[0.6, 0.6, 0.6]])
    bb_min = np.array([[-0.6, -0.6, -0.6]])

    bb_max = v_obs.max(axis=0, keepdims=True)
    bb_min = v_obs.min(axis=0, keepdims=True)

    bb_max = torch.tensor(bb_max, dtype=torch.float32, device='cuda')[0]
    bb_min = torch.tensor(bb_min, dtype=torch.float32, device='cuda')[0]

    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')

    kdtree = torch_kdtree.build_kd_tree(v_obs)

    X_list = []
    Y_list = []
    N_list = []

    X_list, Y_list, N_list = arm_append_list(X_list, Y_list, N_list,
                                             chain, mesh_list, kdtree, v_obs,
                                             numsamples, dim, offset, margin)

    X = torch.cat(X_list, 0)[:numsamples]
    Y = torch.cat(Y_list, 0)[:numsamples]
    N = torch.cat(N_list, 0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    normal = N.detach().cpu().numpy()

    # Speed labels: S*_risk(q) = clip(d_risk(q), d_min=offset, d_max=margin) / margin
    distance0 = distance[:, 0]
    distance1 = distance[:, 1]
    speed = np.zeros((distance.shape[0], 2))
    speed[:, 0] = np.clip(distance0, a_min=offset, a_max=margin) / margin
    speed[:, 1] = np.clip(distance1, a_min=offset, a_max=margin) / margin

    return sampled_points, speed, normal


def point_obstacle_distance(query_points, kdtree, v_obs, normal_obs):

    dists, inds = kdtree.query(query_points, nr_nns_searches=1)
    dists = dists.squeeze()
    inds = inds.squeeze()
    unsigned_distance = torch.sqrt(dists)
    normal_face = normal_obs[inds, :]
    normal = query_points - v_obs[inds, :]
    normal = torch.nn.functional.normalize(normal, dim=1)

    dot = torch.einsum('ij,ij->i', normal, normal_face)
    return unsigned_distance, dot, normal


def point_append_list(X_list, Y_list, N_list,
                      kdtree, v_obs, normal_obs,
                      bb_max, bb_min,
                      numsamples, dim, offset, margin):

    OutsideSize = numsamples + 2
    WholeSize = 0

    while OutsideSize > 0:
        P = 2 * (torch.rand((20 * numsamples, 3), dtype=torch.float32, device='cuda') - 0.5)
        P[:, 0] = P[:, 0] * bb_max[0]
        P[:, 1] = P[:, 1] * bb_max[1]
        P[:, 2] = P[:, 2] * bb_max[2]

        dP = torch.rand((20 * numsamples, 3), dtype=torch.float32, device='cuda') - 0.5
        rL = (torch.rand((20 * numsamples, 1), dtype=torch.float32, device='cuda')) * np.sqrt(dim)

        if dim == 2:
            P[:, 2] = 0
            dP[:, 2] = 0
        nP = P + torch.nn.functional.normalize(dP, dim=1) * (rL)
        PointsInside = (nP[:, 0] <= bb_max[0]) & (nP[:, 0] >= bb_min[0]) &\
                       (nP[:, 1] <= bb_max[1]) & (nP[:, 1] >= bb_min[1]) &\
                       (nP[:, 2] <= bb_max[2]) & (nP[:, 2] >= bb_min[2])
        PointsInside = (nP[:, 0] <= bb_max[0]) & (nP[:, 0] >= bb_min[0]) &\
                       (nP[:, 1] <= bb_max[1]) & (nP[:, 1] >= bb_min[1]) &\
                       (nP[:, 2] <= bb_max[2]) & (nP[:, 2] >= bb_min[2]) &\
                       (P[:, 0] <= bb_max[0]) & (P[:, 0] >= bb_min[0]) &\
                       (P[:, 1] <= bb_max[1]) & (P[:, 1] >= bb_min[1]) &\
                       (P[:, 2] <= bb_max[2]) & (P[:, 2] >= bb_min[2])

        x0 = P[PointsInside, :]
        x1 = nP[PointsInside, :]

        if(x0.shape[0] <= 1):
            continue

        obs_distance0, dot0, normal0 = point_obstacle_distance(x0, kdtree, v_obs, normal_obs)

        where_d = (dot0 < np.sqrt(2) / 2)
        obs_distance0[where_d] *= -1
        normal0[where_d] *= -1

        obs_distance0 = obs_distance0 - offset
        where_d = (obs_distance0 < margin) & (obs_distance0 > offset)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        n0 = normal0[where_d]

        obs_distance1, dot1, normal1 = point_obstacle_distance(x1, kdtree, v_obs, normal_obs)
        where_d = (dot1 < np.sqrt(2) / 2)
        obs_distance1[where_d] *= -1
        normal1[where_d] *= -1

        obs_distance1 = obs_distance1 - offset

        where_d = (obs_distance1 > -offset)

        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = y0[where_d]
        y1 = obs_distance1[where_d]
        n0 = n0[where_d]
        n1 = normal1[where_d]

        print(x0.shape)

        x = torch.cat((x0[:, :dim], x1[:, :dim]), 1)
        y = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1)), 1)
        n = torch.cat((n0[:, :dim], n1[:, :dim]), 1)

        X_list.append(x)
        Y_list.append(y)
        N_list.append(n)

        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]

        if(WholeSize > numsamples):
            break
    return X_list, Y_list, N_list


def point_rand_sample_bound_points(numsamples, dim,
                                   v_obs, n_obs,
                                   offset, margin):
    numsamples = int(numsamples)

    bb_max = np.array([[0.6, 0.6, 0.6]])
    bb_min = np.array([[-0.6, -0.6, -0.6]])

    bb_max = v_obs.max(axis=0, keepdims=True)
    bb_min = v_obs.min(axis=0, keepdims=True)

    bb_max = torch.tensor(bb_max, dtype=torch.float32, device='cuda')[0]
    bb_min = torch.tensor(bb_min, dtype=torch.float32, device='cuda')[0]

    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    n_obs = torch.tensor(n_obs, dtype=torch.float32, device='cuda')

    kdtree = torch_kdtree.build_kd_tree(v_obs)

    X_list = []
    Y_list = []
    N_list = []

    X_list, Y_list, N_list = point_append_list(X_list, Y_list, N_list, kdtree, v_obs, n_obs,
                                               bb_max, bb_min, numsamples, dim, offset, margin)

    X = torch.cat(X_list, 0)[:numsamples]
    Y = torch.cat(Y_list, 0)[:numsamples]
    N = torch.cat(N_list, 0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    normal = N.detach().cpu().numpy()

    distance0 = distance[:, 0]
    distance1 = distance[:, 1]
    speed = np.zeros((distance.shape[0], 2))
    speed[:, 0] = np.clip(distance0 / margin, a_min=offset / margin, a_max=1)
    speed[:, 1] = np.clip(distance1 / margin, a_min=offset / margin, a_max=1)

    return sampled_points, speed, normal


def sample_speed(path, numsamples, dim):

    try:

        global out_path
        out_path = os.path.dirname(path)
        global path_name
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        print('pp', path)
        global task_name
        task_name = out_path.split('/')[2]
        if task_name == 'arm':
            global end_effect
            with open(out_path + '/dim') as f:
                iter = 0
                for line in f:
                    data = line.split()
                    if iter == 0:
                        dim = int(data[0])
                    else:
                        end_effect = data[0]
                        print(end_effect)
                    iter = iter + 1
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path, file_name + '_scaled.off')
        out_file = out_path + '/sampled_points.npy'

        print(input_file)
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')

        limit = 0.5
        xmin = [-limit] * dim
        xmax = [limit] * dim
        velocity_max = 1

        margin = limit / 10. + 0.015
        offset = margin / 10.0 + 0.002

        v, f = igl.read_triangle_mesh(input_file)
        z = np.zeros(3)

        v_obs = v

        start = timer()

        sampled_points, speed, normal = arm_rand_sample_bound_points(numsamples, dim,
                                                                     v_obs, offset, margin)

        end = timer()

        print(sampled_points.shape)

        print(end - start)

        B = np.random.normal(0, 1, size=(3, 128))

        np.save('{}/sampled_points'.format(out_path), sampled_points)
        np.save('{}/speed'.format(out_path), speed)
        np.save('{}/normal'.format(out_path), normal)
        np.save('{}/B'.format(out_path), B)
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))