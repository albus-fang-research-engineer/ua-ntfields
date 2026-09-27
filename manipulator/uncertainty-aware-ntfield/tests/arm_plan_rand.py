"""
Generate 100 collision-free start/goal configurations near obstacles for a UR5
arm, then plan a path between each pair using MPPI.

Pipeline
--------
1. Build the FK chain + bounding-sphere mesh list (build_chain)
2. Load the obstacle mesh and build a KD-tree over its vertices
3. Repeatedly sample IK-feasible joint configs until we have >= 200 that are
   - near obstacles  (0 < signed_dist - margin < margin)
   - not in collision  (signed_dist > 0  after subtracting the safety offset)
4. Randomly pair them into 100 (start, goal) pairs
5. Run MPPI for each pair; save the trajectory if it succeeds

All distance thresholds mirror the ones used in the data-generation file:
    margin = 0.5 / 10  = 0.05
    offset = margin / 10 = 0.005
"""

import sys, os, math, random
import numpy as np
import torch
import igl
import pytorch_kinematics as pk
import torch_kdtree
from timeit import default_timer as timer

sys.path.append('.')
from models.metric_arm import model_test_metric as md

# ------------------------------------------------------------------ #
# Paths / hyper-parameters
# ------------------------------------------------------------------ #
MODEL_PATH  = './Experiments/UR5'
MESH_NAME   = 'Auburn'
DATA_PATH   = f'./datasets/arm/{MESH_NAME}'
MESH_FILE   = f'./datasets/arm/UR5/realpc_scaled.off'

CHECKPOINT  = ('./Experiments/UR5/arm_04_22_10_30/'
               'Model_Epoch_05000_ValLoss_3.696065e-03.pt')
# CHECKPOINT = ('./Experiments/UR5/arm_04_24_19_38/'
#                 'Model_Epoch_04700_ValLoss_2.905046e-03.pt') # old 2cm
# CHECKPOINT = ('./Experiments/UR5/arm_05_19_16_38/'
#                 'Model_Epoch_05000_ValLoss_2.666241e-03.pt') #2cm backup
# CHECKPOINT  = ('./Experiments/UR5/arm_05_19_14_37/'
#                'Model_Epoch_05000_ValLoss_3.540057e-03.pt') #1cm retuned
#######################     Tuned    ##########################
CHECKPOINT  = ('./Experiments/UR5/arm_04_24_18_56/'
               'Model_Epoch_05000_ValLoss_4.250220e-03.pt') #1cm better

# CHECKPOINT = ('./Experiments/UR5/arm_04_24_19_17/'
#                 'Model_Epoch_05000_ValLoss_3.806535e-03.pt') #0cm \ ours nominal

CHECKPOINT = ('./Experiments/UR5/arm_05_19_19_30/'
                'Model_Epoch_05000_ValLoss_2.971892e-03.pt') #2cm working!!!!!!!

# CHECKPOINT = ('./Experiments/UR5/arm_05_20_08_41/'
#                 'Model_Epoch_05000_ValLoss_3.224642e-03.pt') #baseline

INFLATION  = '2cm' 
NUM_PAIRS   = 100            # how many start/goal pairs to plan
SCALE       = math.pi / 0.5  # joint-space normalisation used throughout

LIMIT       = 0.5
MARGIN      = LIMIT / 10.0          # 0.05  — "near obstacle" threshold
OFFSET      = MARGIN / 10.0         # 0.005 — safety clearance subtracted from raw dist

MPPI_STEPS      = 200
MPPI_SAMPLES    = 50
MPPI_HORIZON    = 5
MPPI_SIGMA      = 0.015
MPPI_WEIGHT_T   = -50.0
MPPI_GOAL_W     = 10.0              # weight on cost_start vs cost_goal
CONV_THRESH     = 0.01              # ||goal - current|| < this → converged

BASE = torch.tensor([[0, -0.5*math.pi, 0.0, -0.5*math.pi, 0.0, 0.0]],
                    dtype=torch.float32, device='cuda')   # robot home offset

OUTPUT_DIR = f'Evaluations/Arm/paths_{INFLATION}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ee_path_length(path_rad: np.ndarray, chain) -> float:
    """Sum of EE Cartesian displacements (metres) along the trajectory."""
    q = torch.tensor(path_rad, dtype=torch.float32, device='cuda')
    tg = chain.forward_kinematics(q, end_only=True)
    ee = tg.get_matrix()[:, :3, 3]          # (T, 3)
    return (ee[1:] - ee[:-1]).norm(dim=1).sum().item()

def cspace_path_length(path_rad: np.ndarray) -> float:
    """Sum of joint-space step norms (radians) along the trajectory."""
    q = torch.tensor(path_rad, dtype=torch.float32, device='cuda')
    return (q[1:] - q[:-1]).norm(dim=1).sum().item()
# ================================================================== #
#  1. Build FK chain + bounding-sphere mesh list
# ================================================================== #
def build_chain():
    out_path  = 'datasets/arm/UR5'
    end_link  = 'wrist_3_link'
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    chain = pk.build_serial_chain_from_urdf(
        open(f'{out_path}/ur5e.urdf').read(), end_link)
    chain = chain.to(dtype=torch.float32, device=device)

    # warm-up FK to enumerate link names
    dummy = torch.rand(1, 6, device=device)
    tg    = chain.forward_kinematics(dummy, end_only=False)

    mesh_list = []
    for idx, link_name in enumerate(tg):
        if 2 <= idx < 8:
            arr = np.load(f'{out_path}/meshes/sphere/sphere/{link_name}.npy')
            mesh_list.append(torch.tensor(arr, dtype=torch.float32, device=device))

    return chain, mesh_list


# ================================================================== #
#  2. Forward kinematics → sphere centers in world frame
# ================================================================== #
def FK_spheres(joint_angles_rad: torch.Tensor,
               chain, mesh_list) -> torch.Tensor:
    """
    joint_angles_rad : (N, 6)  — raw radians
    Returns           : (N, M, 4)  — sphere centers (xyz) + radii
    """
    tg_batch = chain.forward_kinematics(joint_angles_rad, end_only=False)
    p_list   = []
    for idx, link_name in enumerate(tg_batch):
        if 2 <= idx < 8:
            balls = mesh_list[idx - 2]          # (M, 4): xyz + radius
            ones  = torch.ones(balls.shape[0], 1, device='cuda')
            nv    = torch.cat([balls[:, :3], ones], dim=1)   # (M, 4)
            m     = tg_batch[link_name].get_matrix()          # (N, 4, 4)
            p     = torch.bmm(m, nv.T.unsqueeze(0).expand(m.shape[0], -1, -1))
            p     = p.permute(0, 2, 1)                        # (N, M, 4)
            p[..., 3] = balls[:, 3]                           # overwrite w with radius
            p_list.append(p)
    return torch.cat(p_list, dim=1)                           # (N, total_spheres, 4)


# ================================================================== #
#  3. Arm–obstacle signed distance for a batch of configs
# ================================================================== #
def arm_collision_distance(joint_batch_norm: torch.Tensor,
                           chain, mesh_list, kdtree) -> torch.Tensor:
    """
    joint_batch_norm : (N, 6) normalised configs  (÷ SCALE)
    Returns          : (N,)  minimum signed distance to obstacle surface
                       positive → free, negative → in collision
    """
    joint_rad = joint_batch_norm * SCALE
    spheres   = FK_spheres(joint_rad, chain, mesh_list)   # (N, S, 4)
    N, S, _   = spheres.shape
    flat      = spheres.reshape(-1, 4)                    # (N*S, 4)

    dists, _  = kdtree.query(flat[:, :3], nr_nns_searches=1)
    dists     = dists.squeeze()                           # (N*S,)
    signed    = torch.sqrt(dists) - flat[:, 3]            # dist_to_surface − sphere_radius

    signed    = signed.reshape(N, S)
    min_dist, _ = signed.min(dim=1)                       # (N,) worst-case sphere
    return min_dist


# ================================================================== #
#  4. Sample IK-feasible configs near obstacles but collision-free
# ================================================================== #
def sample_free_near_obstacle_configs(n_needed: int,
                                      chain, mesh_list, kdtree,
                                      batch_size: int = 4000) -> torch.Tensor:
    """
    Returns (n_needed, 6) normalised joint configs that satisfy:
        signed_dist - OFFSET  ∈ (0, MARGIN)
    i.e. the arm is close to the obstacle surface but NOT in collision.
    """
    collected = []
    total     = 0

    while total < n_needed:
        # --- sample Cartesian EE targets and solve IK ---
        from dataprocessing.torch_IK_UR5 import torch_IK_UR5  # local to your project

        P  = torch.rand((batch_size, 3), dtype=torch.float32, device='cuda')
        P[:, 0] = P[:, 0] * 1.2 + 0.2          # x ∈ [0.2, 1.4]
        P[:, 1] = (P[:, 1] - 0.5) * 1.2         # y ∈ [-0.6, 0.6]
        P[:, 2] = P[:, 2] * 1.2 - 0.1           # z ∈ [-0.1, 1.1]

        a = -0.5*math.pi + (torch.rand(batch_size, 1, device='cuda') - 0.5) * 0.6 * math.pi
        b =                 (torch.rand(batch_size, 1, device='cuda') - 0.5) * 0.6 * math.pi
        c = -0.5*math.pi + (torch.rand(batch_size, 1, device='cuda') - 0.5) * 0.6 * math.pi

        # Build 4×4 end-effector pose matrices
        def _rot(x, a, b, c):
            R = torch.zeros(x.shape[0], 4, 4, dtype=torch.float32, device='cuda')
            ca, sa = torch.cos(a.squeeze(1)), torch.sin(a.squeeze(1))
            cb, sb = torch.cos(b.squeeze(1)), torch.sin(b.squeeze(1))
            cc, sc = torch.cos(c.squeeze(1)), torch.sin(c.squeeze(1))
            R[:, 0, 0] = cb*cc
            R[:, 0, 1] = sa*sb*cc - ca*sc
            R[:, 0, 2] = ca*sb*cc + sa*sc
            R[:, 1, 0] = cb*sc
            R[:, 1, 1] = sa*sb*sc + ca*cc
            R[:, 1, 2] = ca*sb*sc - sa*cc
            R[:, 2, 0] = -sb
            R[:, 2, 1] = sa*cb
            R[:, 2, 2] = ca*cb
            R[:, :3, 3] = x
            R[:,  3, 3] = 1.0
            return R

        poses    = _rot(P, a, b, c)
        torch_ik = torch_IK_UR5(poses.shape[0])
        torch_ik.setJointLimits(-math.pi, math.pi)
        sols     = torch_ik.solveIK(poses)            # (N, K, 6)
        configs  = sols.reshape(-1, 6) / SCALE        # normalised

        # Keep only configs inside the joint-space cube
        inside   = torch.all(configs.abs() <= 0.5, dim=1)
        configs  = configs[inside]

        if configs.shape[0] == 0:
            continue

        # Evaluate obstacle distance
        dist = arm_collision_distance(configs, chain, mesh_list, kdtree)
        dist = dist - OFFSET                          # subtract safety margin

        keep = (dist > 0) & (dist < MARGIN)
        keep = (dist > 0.032) & (dist < 0.09)
        keep = (dist > 0.05) & (dist < 0.09)
        # keep = (dist > 0.08) & (dist < 0.12)
        configs = configs[keep]

        if configs.shape[0] == 0:
            continue

        collected.append(configs.detach())
        total += configs.shape[0]
        print(f'  Collected {total}/{n_needed} near-obstacle configs …')

    all_configs = torch.cat(collected, dim=0)
    return all_configs[:n_needed]


# ================================================================== #
#  5. MPPI path planner
# ================================================================== #
def mppi(model, XP: torch.Tensor):
    """
    XP : (1, 12)  — [start(6) | goal(6)]  normalised joint coords
    Returns (list_of_waypoints, iters_used)
    """
    dP_prior = torch.zeros(1, 6, device='cuda')
    trajectory = [XP[:, :6].clone()]

    for step in range(MPPI_STEPS):
        XP_tmp = XP.clone().unsqueeze(0).repeat(MPPI_SAMPLES, MPPI_HORIZON, 1)

        # Sample perturbations
        noise = (MPPI_SIGMA * torch.randn(MPPI_SAMPLES, 1, 6, device='cuda') +
                 MPPI_SIGMA * torch.randn(MPPI_SAMPLES, MPPI_HORIZON, 6, device='cuda'))
        noise = noise + 2 * dP_prior
        norm  = torch.norm(noise, dim=2, keepdim=True)
        noise = noise / (torch.clamp(norm, min=MPPI_SIGMA) / MPPI_SIGMA)

        cumulative = torch.cumsum(noise, dim=1)
        XP_tmp[..., :6] += cumulative

        # Evaluate cost at first and last horizon step
        endpoints = XP_tmp[:, [0, -1], :].reshape(-1, 12)
        cost      = model.function.TravelTimes(endpoints).reshape(-1, 2)

        combined  = MPPI_GOAL_W * cost[:, 0] + cost[:, 1]

        weights   = torch.softmax(MPPI_WEIGHT_T * combined, dim=0)
        dP_prior  = (weights @ noise[:, 0, :])          # (1, 6)

        XP[:, :6] += dP_prior

        dist_to_goal = torch.norm(XP[:, 6:] - XP[:, :6])
        trajectory.append(XP[:, :6].clone())

        if dist_to_goal < CONV_THRESH:
            break

    trajectory.append(XP[:, 6:].clone())
    return trajectory, step


# ================================================================== #
#  Main
# ================================================================== #
def main():
    # ── reproducibility ──────────────────────────────────────────────
    SEED = 36
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # optional: makes CuDNN deterministic (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print('=== Loading model …')
    model = md.Model(MODEL_PATH, DATA_PATH, 6,
                     [0]*6, device='cuda')
    model.load(CHECKPOINT)
    model.network.eval()

    print('=== Building FK chain …')
    chain, mesh_list = build_chain()

    print('=== Loading obstacle mesh …')
    v, _ = igl.read_triangle_mesh(MESH_FILE)
    v_obs = torch.tensor(v, dtype=torch.float32, device='cuda')
    kdtree = torch_kdtree.build_kd_tree(v_obs)

    # ---------------------------------------------------------------- #
    # Sample 2×NUM_PAIRS valid configs  (half start, half goal)
    # ---------------------------------------------------------------- #
    print(f'=== Sampling {2 * NUM_PAIRS} near-obstacle configs …')
    configs = sample_free_near_obstacle_configs(
        2 * NUM_PAIRS, chain, mesh_list, kdtree
    )
    # Randomly shuffle and split into starts and goals
    perm    = torch.randperm(configs.shape[0])
    starts  = configs[perm[:NUM_PAIRS]]   # (100, 6)  normalised
    goals   = configs[perm[NUM_PAIRS:]]   # (100, 6)  normalised

    print(f'Got {starts.shape[0]} start configs and {goals.shape[0]} goal configs.')

    # ---------------------------------------------------------------- #
    # Plan a path for each pair
    # ---------------------------------------------------------------- #
    success_count = 0
    results = []

    for i in range(NUM_PAIRS):
        # Pack as (1, 12): [start | goal]
        XP = torch.cat([starts[i:i+1], goals[i:i+1]], dim=1).clone()

        # Warm-start CUDA + sync so the timer only measures planning work
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = timer()
        with torch.no_grad():
            traj, iters = mppi(model, XP)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = timer() - t0

        waypoints = torch.cat(traj).cpu().numpy()          # (T, 6) normalised
        waypoints_rad = waypoints * SCALE                  # convert to radians

        final_dist = float(torch.norm(goals[i] - traj[-2][0, :]).cpu())
        success    = final_dist < CONV_THRESH

        if success:
            success_count += 1

        status = 'OK' if success else 'FAIL'
        print(f'  [{i+1:03d}/{NUM_PAIRS}]  iters={iters+1:3d}  '
              f'dist={final_dist:.4f}  t={elapsed:.2f}s  [{status}]')

        results.append({
            'pair_idx':       i,
            'start_norm':     starts[i].cpu().numpy(),
            'goal_norm':      goals[i].cpu().numpy(),
            'path_norm':      waypoints,
            'path_rad':       waypoints_rad,
            'success':        success,
            'iters':          iters + 1,
            'final_dist':     final_dist,
            'planning_time':  elapsed,
        })

        out_path = os.path.join(OUTPUT_DIR, f'path_{i:03d}.npy')
        # np.save(out_path, waypoints_rad)
        if success:
            out_path = os.path.join(OUTPUT_DIR, f'path_{i:03d}.npy')
            np.save(out_path, waypoints_rad)
    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print(f'\n=== Done. Success: {success_count}/{NUM_PAIRS} '
          f'({100*success_count/NUM_PAIRS:.1f}%)')

    # ---------------------------------------------------------------- #
    # Collision rate over finished (converged) trajectories
    # ---------------------------------------------------------------- #
    finished = [r for r in results if r['success']]
    print(f'\n=== Computing collision rate on {len(finished)} finished trajectories …')

    total_waypoints   = 0
    colliding_waypoints = 0
    colliding_paths   = 0

    clean_results = []   # successful + zero collisions

    for i, r in enumerate(finished):
        waypoints = torch.tensor(r['path_norm'], dtype=torch.float32, device='cuda')

        dist = arm_collision_distance(waypoints, chain, mesh_list, kdtree)
        in_collision = dist <= 0

        n_col = int(in_collision.sum().item())
        colliding_waypoints += n_col
        total_waypoints     += waypoints.shape[0]

        if n_col > 0:
            colliding_paths += 1
            col_indices = in_collision.nonzero(as_tuple=True)[0].tolist()
            print(f'  Trajectory {i:03d} (pair index {r["pair_idx"]:03d}): '
                f'{n_col} colliding waypoint(s) at steps {col_indices}')
            col_path = os.path.join(OUTPUT_DIR, f'path_{r["pair_idx"]:03d}_collision.npy')
            np.save(col_path, r['path_rad'])
        else:
            # ── clean trajectory: compute and store lengths ──────────────
            ee_len = ee_path_length(r['path_rad'], chain)
            cs_len = cspace_path_length(r['path_rad'])
            clean_results.append({
                'pair_idx':       r['pair_idx'],
                'ee_length':      ee_len,
                'cs_length':      cs_len,
                'iters':          r['iters'],
                'planning_time':  r['planning_time'],
            })
            print(f'  Trajectory {i:03d} (pair index {r["pair_idx"]:03d}): '
                f'clean  EE={ee_len:.4f}m  CS={cs_len:.4f}rad  t={r["planning_time"]:.2f}s')

    waypoint_collision_rate = colliding_waypoints / total_waypoints
    path_collision_rate     = colliding_paths     / len(finished)

    print(f'  Finished trajectories : {len(finished)}')
    print(f'  Total waypoints       : {total_waypoints}')
    print(f'  Colliding waypoints   : {colliding_waypoints}')
    print(f'  Waypoint collision rate: {100 * waypoint_collision_rate:.2f}%')
    print(f'  Path collision rate    : {100 * path_collision_rate:.2f}%  '
        f'(paths with ≥1 colliding waypoint)')
    if clean_results:
        pair_indices = np.array([c['pair_idx']        for c in clean_results])
        ee_lengths   = np.array([c['ee_length']       for c in clean_results])
        cs_lengths   = np.array([c['cs_length']       for c in clean_results])
        clean_times  = np.array([c['planning_time']   for c in clean_results])

        np.save(os.path.join(OUTPUT_DIR, 'clean_pair_indices.npy'),    pair_indices)
        np.save(os.path.join(OUTPUT_DIR, 'clean_ee_lengths.npy'),      ee_lengths)
        np.save(os.path.join(OUTPUT_DIR, 'clean_cs_lengths.npy'),      cs_lengths)
        np.save(os.path.join(OUTPUT_DIR, 'clean_planning_times.npy'),  clean_times)

        print(f'\n=== Clean trajectories (success + zero collision): {len(clean_results)}')
        print(f'  EE  path length  — mean: {ee_lengths.mean():.4f}m   '
            f'std: {ee_lengths.std():.4f}m   '
            f'median: {np.median(ee_lengths):.4f}m')
        print(f'  C-space length   — mean: {cs_lengths.mean():.4f}rad  '
            f'std: {cs_lengths.std():.4f}rad  '
            f'median: {np.median(cs_lengths):.4f}rad')
        print(f'  Planning time    — mean: {clean_times.mean():.3f}s   '
            f'std: {clean_times.std():.3f}s   '
            f'median: {np.median(clean_times):.3f}s')

    # Save all configs, success mask, and planning times for downstream use
    all_starts      = np.stack([r['start_norm']    for r in results])
    all_goals       = np.stack([r['goal_norm']     for r in results])
    successes       = np.array([r['success']       for r in results])
    planning_times  = np.array([r['planning_time'] for r in results], dtype=np.float64)
    iters_per_pair  = np.array([r['iters']         for r in results], dtype=np.int32)

    np.save(os.path.join(OUTPUT_DIR, 'starts_norm.npy'),     all_starts)
    np.save(os.path.join(OUTPUT_DIR, 'goals_norm.npy'),      all_goals)
    np.save(os.path.join(OUTPUT_DIR, 'successes.npy'),       successes)
    np.save(os.path.join(OUTPUT_DIR, 'planning_times.npy'),  planning_times)
    np.save(os.path.join(OUTPUT_DIR, 'iters_per_pair.npy'),  iters_per_pair)

    # Planning-time summary (over successful pairs, since failures hit MPPI_STEPS)
    if success_count > 0:
        succ_mask  = successes
        succ_times = planning_times[succ_mask]
        print(f'\n=== Planning time (over {success_count} successful pairs) ===')
        print(f'  mean:   {succ_times.mean():.3f}s')
        print(f'  std:    {succ_times.std():.3f}s')
        print(f'  median: {np.median(succ_times):.3f}s')
        print(f'  min:    {succ_times.min():.3f}s')
        print(f'  max:    {succ_times.max():.3f}s')
    print(f'\n=== Planning time (all {NUM_PAIRS} pairs, incl. failures) ===')
    print(f'  mean:   {planning_times.mean():.3f}s')
    print(f'  std:    {planning_times.std():.3f}s')
    print(f'  median: {np.median(planning_times):.3f}s')

    # Start/goal pairs for successful (converged) trajectories
    successful_pairs_rad = np.stack([
        np.stack([r['start_norm'] * SCALE, r['goal_norm'] * SCALE])
        for r in results if r['success']
    ])  # shape: (N_success, 2, 6) — radians
    success_file_name = 'map2_mppi_success.npy'
    np.save(os.path.join(OUTPUT_DIR, success_file_name), successful_pairs_rad)
    print(f'Saved {len(successful_pairs_rad)} start/goal pairs to {success_file_name}')
    print(f'Results saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()