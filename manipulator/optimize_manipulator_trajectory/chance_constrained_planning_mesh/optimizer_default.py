"""
solve_step_ur5e.py  (mesh-inference version)

Key change: mu_sigma_grad_nn(...) replaced by mu_sigma_grad_mesh(...).
The `model` argument is now a MeshDistanceModel, and `device` is no longer
needed here (it lives inside the model).  The `rollout_optimized` caller
passes the same model object every step, so chain + KD-tree are built once.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mesh_inference.mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh, SIGMA_CONST

URDF_PATH = "/manipulator_simulator/src/ur5e_rgbd/urdf/ur5e_rgbd.urdf"
URDF_PATH = "/workspace/datasets/arm/UR5/ur5e.urdf"
# EE_FRAME  = pin_model.getFrameId("tool0")

BETA = 2.0   # keep the same safety level as before; tune as needed

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

    N_VALID = 16  # or whatever constant used in mesh_inference.py

    for i, q in enumerate(traj_rad):
        mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q, mesh_model)

        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        risk = mu - beta * sigma

        k = int(np.argmin(mu))          # stacked constraint index
        link_idx = k // N_VALID         # recover original link index

        print(
            f"{i:02d} | "
            f"{mu[k]: .6f} | "
            f"{risk[k]: .6f} | "
            f"{LINK_NAMES[link_idx]} | "
            f"{k}"
        )
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


# ── Main solver step ──────────────────────────────────────────────────────────

def solve_step_ur5e(
    q0,
    q_goal,
    mesh_model: MeshDistanceModel,      # replaces (model, device)
    epoch: int,
    folder: str = "",
    safety_margin: float = 0.0,
    verbose: bool = False,
):
    """
    One chance-constrained IK step using sphere-mesh distance instead of NN.

    Parameters
    ----------
    q0, q_goal  : (6,) joint configs
    mesh_model  : MeshDistanceModel  (built once outside the loop, holds obstacle cloud)
    epoch           : for debug naming
    folder          : debug plot output dir (empty = no plots)
    safety_margin   : extra deterministic clearance [m]
    verbose         : print debug info

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
    # q0     = wrap_to_pi(np.asarray(q0,     dtype=np.float64).reshape(6))
    # q_goal = wrap_to_pi(np.asarray(q_goal, dtype=np.float64).reshape(6))

    
    # Q        = np.diag([30.0]*6)
    # D        = np.diag([100., 100., 100., 20., 20., 20.])
    Q = np.diag([150., 150., 150., 20., 20., 20.])
    D = np.diag([80., 80., 80., 20., 20., 20.])
    Q = np.diag([100., 100., 100., 20., 20., 20.])
    D = np.diag([550., 550., 550., 20., 20., 20.])
    # Q = np.diag([120., 120., 120., 20., 20., 20.])
    # D = np.diag([150., 150., 150., 40., 40., 40.])
    # D = np.diag([5., 5., 5., 1., 1., 1.])
    dq_limit = 0.15   # rad/step

    # ── FK + Jacobian ─────────────────────────────────────────────────────
    x_curr = fk_ur5e(q0)
    x_goal = fk_ur5e(q_goal)
    J      = jacobian_ur5e(q0)

    # ── Mesh-based distance, gradient, constant sigma ─────────────────────
    mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q0, mesh_model)
    # mu   : (N_LINKS,)
    # sigma: (N_LINKS,)  all == SIGMA_CONST (0.02 m)
    # grad : (N_LINKS, 6)

    mu    = np.asarray(mu,    dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    grad  = np.asarray(grad,  dtype=np.float64).reshape(-1, 6)

    if verbose:
        print("\n--- Mesh Chance-Constrained IK ---")
        print("q0          :", q0)
        print("x_curr      :", x_curr)
        print("x_goal      :", x_goal)
        print("min(mu)     :", np.min(mu))
        print("min(mu-β·σ) :", np.min(mu - BETA * sigma))
        print("sigma       :", sigma[0], "(constant)")
        print("J cond num  :", np.linalg.cond(J))

    # ── Optimisation: z = [dq(6), slack(6)] ──────────────────────────────
    def objective(z):
        dq, slack = z[:6], z[6:]
        return dq @ Q @ dq + slack @ D @ slack

    def tracking_constraint_default(z):
        dq, slack = z[:6], z[6:]
        return x_curr + J @ dq - x_goal - slack   # == 0
    def tracking_constraint(z):
        dq, slack = z[:6], z[6:]
        return pose_error_6d(x_curr + J @ dq, x_goal) - slack   # == 0
    def chance_constraints(z):
        dq = z[:6]
        return mu + grad @ dq - BETA * sigma - safety_margin   # >= 0

    q_min = np.full(6, -6.28)
    q_max = np.full(6,  6.28)
    # q_min = np.full(6, -np.pi)
    # q_max = np.full(6,  np.pi)

    def joint_lower(z):
        return (q0 + z[:6]) - q_min

    def joint_upper(z):
        return q_max - (q0 + z[:6])

    cons = [
        {"type": "eq",   "fun": tracking_constraint},
        {"type": "ineq", "fun": chance_constraints},
        {"type": "ineq", "fun": joint_lower},
        {"type": "ineq", "fun": joint_upper},
    ]
    z0 = np.zeros(12, dtype=np.float64)
    dq_nom = wrap_to_pi(q_goal - q0)
    z0[:6] = dq_nom
    res = minimize(
        objective,
        z0,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-9, "disp": verbose},
    )

    if not res.success:
        print("[WARN] Optimisation did not converge:", res.message)
        diagnose_slsqp_failure(
            res=res,
            q0=q0,
            q_goal=q_goal,
            x_curr=x_curr,
            x_goal=x_goal,
            J=J,
            mu=mu,
            sigma=sigma,
            grad=grad,
            safety_margin=safety_margin,
            q_min=q_min,
            q_max=q_max,
        )
    dq_opt    = res.x[:6]
    slack_opt = res.x[6:]
    q_next    = (q0 + dq_opt)
    # q_next    = wrap_to_pi(q0 + dq_opt)

    for i in range(6):
        if abs(dq_opt[i]) > dq_limit:
            print(f"[DEBUG] dq_limit exceeded joint {i}: {dq_opt[i]:.4f}")

    if verbose:
        print("dq_opt  :", dq_opt)
        print("slack   :", slack_opt)
        print("q_next  :", q_next)
        print("chance residual min :", np.min(chance_constraints(res.x)))
        print("tracking residual   :", np.linalg.norm(tracking_constraint(res.x)))

    debug = dict(
        mu=mu, sigma=sigma, grad=grad, obs_k=obs_k,
        J=J, x_curr=x_curr, x_goal=x_goal,
        dq_opt=dq_opt, slack_opt=slack_opt, result=res,
    )

    if folder:
        plot_chance_debug_jointspace(
            q0=q0, q_next=q_next, x_curr=x_curr, x_goal=x_goal,
            mu=mu, sigma=sigma, grad=grad,
            folder=folder, epoch=epoch, step_id=step_id,
        )

    return q_next, debug

def diagnose_slsqp_failure(res, q0, q_goal, x_curr, x_goal, J,
                           mu, sigma, grad, safety_margin, q_min, q_max,
                           beta=BETA):
    z = res.x
    dq = z[:6]
    slack = z[6:]

    tracking_res = x_curr + J @ dq - x_goal - slack
    chance_res = mu + grad @ dq - beta * sigma - safety_margin
    joint_lower_res = (q0 + dq) - q_min
    joint_upper_res = q_max - (q0 + dq)

    grad_norms = np.linalg.norm(grad, axis=1)
    active = np.where(chance_res < 1e-4)[0]
    violated = np.where(chance_res < -1e-6)[0]

    print("\n========== SLSQP DIAGNOSTIC ==========")
    print("success:", res.success)
    print("status :", res.status)
    print("message:", res.message)
    print("nit    :", getattr(res, "nit", None))
    print("fun    :", getattr(res, "fun", None))

    print("\n--- Step size ---")
    print("dq:", dq)
    print("max |dq|:", np.max(np.abs(dq)))
    print("||dq||:", np.linalg.norm(dq))
    print("raw q_goal - q0 wrapped:", wrap_to_pi(q_goal - q0))
    print("max raw waypoint jump:", np.max(np.abs(wrap_to_pi(q_goal - q0))))

    print("\n--- Tracking residual ---")
    print("||tracking_res||:", np.linalg.norm(tracking_res))
    print("tracking_res:", tracking_res)
    print("||slack||:", np.linalg.norm(slack))
    print("slack:", slack)

    print("\n--- Chance constraints ---")
    print("min chance residual:", np.min(chance_res))
    print("num violated:", len(violated))
    print("num active-ish:", len(active))
    print("worst constraint idx:", int(np.argmin(chance_res)))
    print("worst mu:", mu[np.argmin(chance_res)])
    print("worst sigma:", sigma[np.argmin(chance_res)])
    print("worst grad norm:", grad_norms[np.argmin(chance_res)])
    print("violated idx:", violated[:20])

    print("\n--- Initial safety ---")
    initial_risk = mu - beta * sigma - safety_margin
    print("min initial mu:", np.min(mu))
    print("min initial risk margin:", np.min(initial_risk))
    print("initial worst constraint:", int(np.argmin(initial_risk)))

    print("\n--- Gradient sanity ---")
    print("min grad norm:", np.min(grad_norms))
    print("max grad norm:", np.max(grad_norms))
    print("num zero grad rows:", np.sum(grad_norms < 1e-10))
    print("zero grad rows near obstacle:",
          np.where((grad_norms < 1e-10) & (mu < 0.10))[0])

    print("\n--- Joint bounds ---")
    print("min joint lower residual:", np.min(joint_lower_res))
    print("min joint upper residual:", np.min(joint_upper_res))
    print("lower violated joints:", np.where(joint_lower_res < -1e-8)[0])
    print("upper violated joints:", np.where(joint_upper_res < -1e-8)[0])

    print("======================================\n")
# ── Debug plots ───────────────────────────────────────────────────────────────

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