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
import pinocchio as pin
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mesh_inference.mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh, SIGMA_CONST

URDF_PATH = "/manipulator_simulator/src/ur5e_rgbd/urdf/ur5e_rgbd.urdf"
pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data  = pin_model.createData()
EE_FRAME  = pin_model.getFrameId("tool0")

BETA = 2.0   # keep the same safety level as before; tune as needed

debug_step_counter = 0
last_debug_epoch   = -1


# ── Pinocchio helpers ─────────────────────────────────────────────────────────

def fk_ur5e(q):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pose = pin_data.oMf[EE_FRAME]
    rpy  = pin.rpy.matrixToRpy(pose.rotation)
    return np.concatenate([pose.translation, rpy])


def jacobian_ur5e(q):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin.computeFrameJacobian(
        pin_model, pin_data, q, EE_FRAME,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )


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
    verbose: bool = True,
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

    Q        = np.diag([1.0]*6)
    D        = np.diag([100., 100., 100., 20., 20., 20.])
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

    def tracking_constraint(z):
        dq, slack = z[:6], z[6:]
        return x_curr + J @ dq - x_goal - slack   # == 0

    def chance_constraints(z):
        dq = z[:6]
        return mu + grad @ dq - BETA * sigma - safety_margin   # >= 0

    q_min = np.full(6, -6.28)
    q_max = np.full(6,  6.28)

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

    res = minimize(
        objective,
        np.zeros(12, dtype=np.float64),
        method="SLSQP",
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-9, "disp": verbose},
    )

    if not res.success:
        print("[WARN] Optimisation did not converge:", res.message)

    dq_opt    = res.x[:6]
    slack_opt = res.x[6:]
    q_next    = q0 + dq_opt

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