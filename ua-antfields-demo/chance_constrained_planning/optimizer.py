import numpy as np
import quadprog
from scipy.stats import norm
from load_njsdf.inference import mu_sigma_grad_nn, BETA, sigma_nn
import torch
import matplotlib.pyplot as plt

# DELTA = 0.1/10
# BETA = norm.ppf(1 - DELTA) # chance constraint

debug_step_counter = 0
last_debug_epoch = -1


def solve_step(p0, p_goal, obstacle_points, model, device, epoch, path_start, folder=None):
    '''
    p0 is current position, p_goal is the next waypoint to track
    p0 and p_goal are not global start and goal points
    '''
    global debug_step_counter, last_debug_epoch
    if epoch != last_debug_epoch:
        debug_step_counter = 0
        last_debug_epoch = epoch
    step_id = debug_step_counter
    debug_step_counter += 1

    if torch.is_tensor(p0):
        p0 = p0.detach().cpu().numpy()
    if torch.is_tensor(p_goal):
        p_goal = p_goal.detach().cpu().numpy()

    mu, sigma, grad, obs_k = mu_sigma_grad_nn(p0, obstacle_points, model, device)
    # sigma_start = sigma_nn(path_start, obs_k, model, device)
    # sigma = sigma_start

    if folder is not None:
        print("\n--- Chance constraint debug ---")
        print("robot:", p0)
        print("goal:", p_goal)
        print("min(mu):", np.min(mu))
        print("min(risk):", np.min(mu - BETA * sigma))
        print("mean(sigma):", np.mean(sigma))

    # ---- QP (slack eliminated analytically) ----
    # slack = p0[:2] + dp - p_goal[:2]
    # obj   = ||dp||^2 + 100*||slack||^2
    #       = 101*dp@dp + 200*e@dp + 100*||e||^2     ( e = p0[:2] - p_goal[:2] )
    #
    # quadprog form: min 0.5 x^T G x - a^T x   s.t.  C^T x >= b
    e = (p0[:2] - p_goal[:2]).astype(np.float64)
    G = 202.0 * np.eye(2)
    a = -200.0 * e

    # Chance: mu + grad @ dp >= BETA*sigma  =>  C^T dp >= b
    C = np.ascontiguousarray(grad.T, dtype=np.float64)   # (2, K)
    b = (BETA * sigma - mu).astype(np.float64)           # (K,)

    try:
        dp = quadprog.solve_qp(G, a, C, b, meq=0)[0]
    except ValueError:
        # Linearized chance constraints infeasible — stay put.
        dp = np.zeros(2)
        if folder is not None:
            print("WARNING: QP infeasible at step", step_id)

    p_next = p0.copy()
    p_next[:2] += dp

    if folder is not None:
        plot_chance_debug(p0, p_next, p_goal, obstacle_points,
                          mu, sigma, grad, obs_k, folder, epoch, step_id)

    return p_next, mu, sigma


def plot_chance_debug(robot_xy, p_next, unoptimized_waypoint, obstacle_points,
                      mu, sigma, grad, obs_k, folder, epoch, step_id):
    import os

    epoch_folder = os.path.join(folder, f"epoch_{epoch:04d}")
    os.makedirs(epoch_folder, exist_ok=True)

    # convert tensors to numpy if needed
    if torch.is_tensor(obstacle_points):
        obstacle_points = obstacle_points.detach().cpu().numpy()
    if torch.is_tensor(p_next):
        p_next = p_next.detach().cpu().numpy()
    if torch.is_tensor(unoptimized_waypoint):
        unoptimized_waypoint = unoptimized_waypoint.detach().cpu().numpy()
    if torch.is_tensor(obs_k):
        obs_k = obs_k.detach().cpu().numpy()
    if torch.is_tensor(robot_xy):
        robot_xy = robot_xy.detach().cpu().numpy()

    # ---------- Plot 1: gradient directions ----------
    plt.figure(figsize=(6, 6))

    plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1],
                s=2, alpha=0.15, label="all obstacles")
    plt.scatter(obs_k[:, 0], obs_k[:, 1],
                s=20, c="orange", label="K constraints")
    plt.scatter(robot_xy[0], robot_xy[1],
                c='red', s=50, label="robot")
    plt.scatter(p_next[0], p_next[1],
                c='green', s=20, label="optimized")
    plt.scatter(unoptimized_waypoint[0], unoptimized_waypoint[1],
                c='purple', s=20, label="nominal waypoint")

    plt.plot([robot_xy[0], unoptimized_waypoint[0]],
             [robot_xy[1], unoptimized_waypoint[1]],
             linestyle="--", color="purple", linewidth=2)
    plt.plot([robot_xy[0], p_next[0]],
             [robot_xy[1], p_next[1]],
             color="green", linewidth=2)

    for i in range(len(mu)):
        g = grad[i]
        plt.arrow(robot_xy[0], robot_xy[1],
                  g[0] * 0.02, g[1] * 0.02,
                  head_width=0.002, color="blue", alpha=0.7)

    points = np.vstack([
        robot_xy[:2],
        p_next[:2],
        unoptimized_waypoint[:2],
        obs_k[:, :2]
    ])
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    padding = 0.05
    plt.xlim(xmin - padding, xmax + padding)
    plt.ylim(ymin - padding, ymax + padding)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Chance constraint gradients")
    plt.savefig(f"{epoch_folder}/step_{step_id:05d}_grad.png")
    plt.close()