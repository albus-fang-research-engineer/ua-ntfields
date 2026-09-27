"""
solve_step_ur5e.py  (QP version, slack eliminated)

solve_step_ur5e replaces SLSQP with a strictly convex QP (slack eliminated
analytically), solved via quadprog. All helper functions preserved.
"""

import os
import numpy as np
import quadprog
import torch
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mesh_inference.mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh#, SIGMA_CONST

URDF_PATH = "/manipulator_simulator/src/ur5e_rgbd/urdf/ur5e_rgbd.urdf"
URDF_PATH = "/workspace/datasets/arm/UR5/ur5e.urdf"

from scipy.stats import norm

DELTA_J = 0.2            # declared joint risk per step
K_LINKS = 6
N_VAL = 10              
KR      = K_LINKS * N_VAL

BETA = norm.ppf(1.0 - DELTA_J / KR)   # ≈ 2.71 for Δ_j=0.2, KR=60

debug_step_counter = 0
last_debug_epoch   = -1

import roboticstoolbox as rtb

robot = rtb.ERobot.URDF(URDF_PATH)
EE_LINK = "tool0"
LINK_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
print_sensor_std = False

# ── Debug printers ──────────────────────────────────────────────────────────

def print_waypoint_distances_old(traj_rad, mesh_model, beta=2.0):
    print("\nWaypoint distance report")
    print("idx | min(mu) | min(mu-beta*sigma) | closest link")
    print("----+---------+--------------------+-------------")

    for i, q in enumerate(traj_rad):
        mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q, mesh_model)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        risk = mu - beta * sigma
        k = int(np.argmin(mu))
        print(
            f"{i:02d} | "
            f"{np.min(mu): .6f} | "
            f"{np.min(risk): .6f} | "
            f"{LINK_NAMES[k]}"
        )


def print_waypoint_distances(traj_rad, mesh_model, beta=2.0):
    print("\nWaypoint distance report")
    print("idx | min(mu) | min(mu-beta*sigma) | closest link | constraint")
    print("----+---------+--------------------+--------------+-----------")

    N_VALID = 16  # match the constant used in mesh_inference.py

    for i, q in enumerate(traj_rad):
        mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q, mesh_model)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        risk = mu - beta * sigma
        k = int(np.argmin(mu))
        link_idx = k // N_VALID
        print(
            f"{i:02d} | "
            f"{mu[k]: .6f} | "
            f"{risk[k]: .6f} | "
            f"{LINK_NAMES[link_idx]} | "
            f"{k}"
        )


# ── Math helpers ────────────────────────────────────────────────────────────

def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi


def fk_ur5e(q):
    T = robot.fkine(q, end=EE_LINK)
    rpy = T.rpy(order='zyx')
    return np.concatenate([T.t, rpy])


def jacobian_ur5e(q):
    return robot.jacob0_analytical(q, representation='rpy/zyx', end=EE_LINK)


def wrap_angle_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def pose_error_6d(x_curr, x_goal):
    pos_err = x_curr[:3] - x_goal[:3]
    rot_err = wrap_angle_diff(x_curr[3:], x_goal[3:])
    return np.concatenate([pos_err, rot_err])


# ── Main solver step (QP, slack eliminated) ────────────────────────────────

def solve_step_ur5e(
    q0,
    q_goal,
    mesh_model: MeshDistanceModel,
    epoch: int,
    folder: str = "",
    safety_margin: float = 0.0,
    verbose: bool = False,
):
    """
    One chance-constrained IK step using sphere-mesh distance.
    Slack absorbs pose-tracking error; QP solved with quadprog.

    Parameters
    ----------
    q0, q_goal    : (6,) joint configs
    mesh_model    : MeshDistanceModel (built once outside the loop)
    epoch         : for debug naming
    folder        : debug plot output dir (empty = no plots)
    safety_margin : extra deterministic clearance [m]
    verbose       : print debug info

    Returns
    -------
    q_next : (6,)
    debug  : dict
    """
    global debug_step_counter, last_debug_epoch

    if epoch != last_debug_epoch:
        debug_step_counter = 0
        last_debug_epoch   = epoch
    step_id = debug_step_counter
    debug_step_counter += 1

    # ── Normalise inputs ──────────────────────────────────────────────────
    if torch.is_tensor(q0):
        q0 = q0.detach().cpu().numpy()
    if torch.is_tensor(q_goal):
        q_goal = q_goal.detach().cpu().numpy()
    q0     = np.asarray(q0,     dtype=np.float64).reshape(6)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)

    # ── Weights and limits ────────────────────────────────────────────────
    Q = np.diag([100., 100., 100.,  20.,  20.,  20.])
    D = np.diag([550., 550., 550.,  20.,  20.,  20.])
    dq_limit = 0.15

    # ── FK + Jacobian ─────────────────────────────────────────────────────
    x_curr = fk_ur5e(q0)
    x_goal = fk_ur5e(q_goal)
    J      = jacobian_ur5e(q0)

    # ── Mesh-based distance, gradient, constant sigma ─────────────────────
    mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q0, mesh_model)
    mu    = np.asarray(mu,    dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    grad  = np.asarray(grad,  dtype=np.float64).reshape(-1, 6)

    # ── Obstacle points actually used in the constraints this step ────────
    # (always printed) — d recovered from sigma since sigma = sigma_coeff * d^2
    if print_sensor_std:
        d_used = np.sqrt(sigma / mesh_model._sigma_coeff)

        idx     = torch.as_tensor(obs_k, dtype=torch.long, device=mesh_model.device)
        obs_xyz = mesh_model._obs_pts[idx].detach().cpu().numpy()      # (K, 3)

        n_used   = len(obs_k)
        n_unique = len(np.unique(obs_k))
        n_cloud  = mesh_model._obs_pts.shape[0]
        print(f"\n  Constraint points: {n_used} used "
            f"({n_unique} unique) of {n_cloud} in cloud")
        print(f"  {'#':>3} {'obs_k':>7} {'d[m]':>7} {'sigma':>7}    obstacle xyz")

        order = np.argsort(d_used)            # closest first = the binding ones
        for c in order:
            p = obs_xyz[c]
            print(f"  {c:3d} {obs_k[c]:7d} {d_used[c]:7.3f} {sigma[c]:7.4f}    "
                f"[{p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}]")
    if verbose:
        print("\n--- Mesh Chance-Constrained IK (QP) ---")
        print("q0          :", q0)
        print("x_curr      :", x_curr)
        print("x_goal      :", x_goal)
        print("min(mu)     :", np.min(mu))
        print("min(mu-β·σ) :", np.min(mu - BETA * sigma))
        print("sigma       :", sigma[0])
        print("J cond num  :", np.linalg.cond(J))

    # ── Slack-eliminated QP in Δq ─────────────────────────────────────────
    # slack = pose_error_6d(x_curr + J Δq, x_goal) ≈ e + J Δq
    # objective: Δq^T Q Δq + slack^T D slack
    #          = Δq^T (Q + J^T D J) Δq + 2 (J^T D e)^T Δq + const
    # quadprog form: min 0.5 x^T G x - a^T x  s.t. C^T x >= b
    e = pose_error_6d(x_curr, x_goal)

    G_qp = 2.0 * (Q + J.T @ D @ J)
    a_qp = -2.0 * (J.T @ D @ e)

    q_min = np.full(6, -6.28)
    q_max = np.full(6,  6.28)

    # Stack constraints as rows, then transpose for quadprog's (n_vars, m) shape.
    C = np.vstack([
        grad,          # (K, 6)  chance:        grad Δq >= β σ + s_margin - μ
        np.eye(6),     # (6, 6)  joint lower:   Δq      >= q_min - q0
        -np.eye(6),    # (6, 6)  joint upper:  -Δq      >= q0 - q_max
    ]).T

    b = np.concatenate([
        BETA * sigma + safety_margin - mu,
        q_min - q0,
        q0 - q_max,
    ])

    G_qp = np.ascontiguousarray(G_qp, dtype=np.float64)
    a_qp = np.ascontiguousarray(a_qp, dtype=np.float64)
    C    = np.ascontiguousarray(C,    dtype=np.float64)
    b    = np.ascontiguousarray(b,    dtype=np.float64)

    qp_status = "ok"
    try:
        sol    = quadprog.solve_qp(G_qp, a_qp, C, b, meq=0)
        dq_opt = sol[0]
    except ValueError as err:
        qp_status = f"infeasible: {err}"
        if verbose:
            print(f"[WARN] QP infeasible at this step; Δq = 0 ({err})")
        dq_opt = np.zeros(6)

    q_next    = q0 + dq_opt
    slack_opt = e + J @ dq_opt

    for i in range(6):
        if abs(dq_opt[i]) > dq_limit:
            print(f"[DEBUG] dq_limit exceeded joint {i}: {dq_opt[i]:.4f}")

    if verbose:
        print("dq_opt :", dq_opt)
        print("slack  :", slack_opt)
        print("q_next :", q_next)
        chance_res = mu + grad @ dq_opt - BETA * sigma - safety_margin
        print("chance residual min :", np.min(chance_res))
        print("tracking residual   :", np.linalg.norm(slack_opt))

    debug = dict(
        mu=mu, sigma=sigma, grad=grad, obs_k=obs_k,
        J=J, x_curr=x_curr, x_goal=x_goal,
        dq_opt=dq_opt, slack_opt=slack_opt,
        qp_status=qp_status,
    )

    if folder:
        plot_chance_debug_jointspace(
            q0=q0, q_next=q_next, x_curr=x_curr, x_goal=x_goal,
            mu=mu, sigma=sigma, grad=grad,
            folder=folder, epoch=epoch, step_id=step_id,
        )

    return q_next, debug


# ── Failure diagnostics (kept from the SLSQP version, mostly unchanged) ────

def diagnose_qp_failure(dq, q0, q_goal, x_curr, x_goal, J,
                         mu, sigma, grad, safety_margin, q_min, q_max,
                         beta=BETA):
    """
    Manual diagnostic to call after a QP fallback to dq=0 (or whenever you
    want to inspect the linearized problem state at the current step).
    """
    slack = pose_error_6d(x_curr + J @ dq, x_goal)

    chance_res = mu + grad @ dq - beta * sigma - safety_margin
    joint_lower_res = (q0 + dq) - q_min
    joint_upper_res = q_max - (q0 + dq)

    grad_norms = np.linalg.norm(grad, axis=1)
    active = np.where(chance_res < 1e-4)[0]
    violated = np.where(chance_res < -1e-6)[0]

    print("\n========== QP STEP DIAGNOSTIC ==========")
    print("--- Step size ---")
    print("dq:", dq)
    print("max |dq|:", np.max(np.abs(dq)))
    print("||dq||:", np.linalg.norm(dq))
    print("raw q_goal - q0 wrapped:", wrap_to_pi(q_goal - q0))

    print("\n--- Tracking residual ---")
    print("||slack||:", np.linalg.norm(slack))
    print("slack:", slack)

    print("\n--- Chance constraints ---")
    print("min chance residual:", np.min(chance_res))
    print("num violated:", len(violated))
    print("num active-ish:", len(active))
    if len(chance_res):
        worst = int(np.argmin(chance_res))
        print("worst constraint idx:", worst)
        print("worst mu:", mu[worst])
        print("worst sigma:", sigma[worst])
        print("worst grad norm:", grad_norms[worst])

    print("\n--- Initial safety ---")
    initial_risk = mu - beta * sigma - safety_margin
    print("min initial mu:", np.min(mu))
    print("min initial risk margin:", np.min(initial_risk))

    print("\n--- Gradient sanity ---")
    print("min grad norm:", np.min(grad_norms))
    print("max grad norm:", np.max(grad_norms))
    print("num zero grad rows:", np.sum(grad_norms < 1e-10))

    print("\n--- Joint bounds ---")
    print("min joint lower residual:", np.min(joint_lower_res))
    print("min joint upper residual:", np.min(joint_upper_res))
    print("========================================\n")


# ── Debug plots ─────────────────────────────────────────────────────────────

def plot_chance_debug_jointspace(q0, q_next, x_curr, x_goal,
                                  mu, sigma, grad, folder, epoch, step_id):
    epoch_folder = os.path.join(folder, f"epoch_{epoch:04d}")
    os.makedirs(epoch_folder, exist_ok=True)

    risk = mu - BETA * sigma
    idx  = np.arange(len(mu))

    plt.figure(figsize=(8, 4))
    plt.plot(idx, mu,   label="mu")
    plt.plot(idx, risk, label="mu - β·sigma")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Link index")
    plt.ylabel("Distance [m]")
    plt.title("Chance constraint margins (mesh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{epoch_folder}/step_{step_id:05d}_chance_margins.png")
    plt.close()

    grad_norm = np.linalg.norm(grad, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(idx, grad_norm)
    plt.xlabel("Link index"); plt.ylabel("||∂d/∂q||")
    plt.title("Constraint gradient norms")
    plt.tight_layout()
    plt.savefig(f"{epoch_folder}/step_{step_id:05d}_grad_norms.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(6), q0,     'o-', label="q0")
    plt.plot(np.arange(6), q_next, 'o-', label="q_next")
    plt.xlabel("Joint index"); plt.ylabel("Joint value [rad]")
    plt.title("Joint update"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{epoch_folder}/step_{step_id:05d}_joint_update.png")
    plt.close()