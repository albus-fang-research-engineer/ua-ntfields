import torch
import numpy as np
import matplotlib.pyplot as plt
from sdf.stochastic_robot_sdf import RobotSdfCollisionNet

MODEL_PATH = "models/sdf_2d.pt"
DATA_PATH = "../dataset/turtlebot2d_geom.npy"

RADIUS = 0.0105
GT_VAR = 0.002**2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# MODEL
# -------------------------------------------------
def predict_mu_var(model, x):
    pred = model(x)
    mu, logvar = torch.chunk(pred, 2, dim=-1)
    logvar = torch.clamp(logvar, -20.0, 10.0)
    return mu.squeeze(-1), torch.exp(logvar).squeeze(-1)


model = RobotSdfCollisionNet(
    in_channels=4,
    out_channels=1,
    layers=[128] * 4,
    skips=[]
).model

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

print("Model loaded")


# -------------------------------------------------
# ANALYTIC GT
# -------------------------------------------------
def gt_signed_distance(robot_xy, points):
    return np.linalg.norm(points - robot_xy, axis=1) - RADIUS


# -------------------------------------------------
# SPATIAL ERROR
# -------------------------------------------------
NUM_POSES = 4
POINTS_PER_POSE = 4000

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax in axes.flat:

    robot_xy = np.random.uniform(-1.5, 1.5, size=2)

    points = np.random.uniform(-2, 2, size=(POINTS_PER_POSE, 2))
    gt = gt_signed_distance(robot_xy, points)

    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], POINTS_PER_POSE, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        mu, _ = predict_mu_var(
            model,
            torch.tensor(x_input, dtype=torch.float32, device=device)
        )

    pred = mu.cpu().numpy()
    error = np.abs(pred - gt)

    sc = ax.scatter(points[:, 0], points[:, 1], c=error, s=3, cmap="coolwarm")

    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.set_title(f"Robot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

fig.colorbar(sc, ax=axes.ravel().tolist(), label="|Prediction Error| [m]")
plt.savefig("stochastic_plots/spatial_error.png", dpi=200)


# -------------------------------------------------
# SPATIAL VARIANCE
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax in axes.flat:

    robot_xy = np.random.uniform(-1.5, 1.5, size=2)
    points = np.random.uniform(-2, 2, size=(POINTS_PER_POSE, 2))

    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], POINTS_PER_POSE, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        _, var = predict_mu_var(
            model,
            torch.tensor(x_input, dtype=torch.float32, device=device)
        )

    pred_var = var.cpu().numpy()

    sc = ax.scatter(points[:, 0], points[:, 1], c=pred_var, s=3, cmap="viridis")

    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.set_title(f"Predicted σ² @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

fig.colorbar(sc, ax=axes.ravel().tolist(), label="Predicted variance σ²")
plt.savefig("stochastic_plots/spatial_variance.png", dpi=200)


# -------------------------------------------------
# 5 RANDOM POSES (ANALYTIC GT)
# -------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax in axes:

    robot_xy = np.random.uniform(-0.5, 0.5, size=2)
    point = np.random.uniform(-0.36, 0.36, size=(1, 2))

    gt = gt_signed_distance(robot_xy, point)[0]

    x_input = np.concatenate([robot_xy[None, :], point], axis=1)

    with torch.no_grad():
        mu, var = predict_mu_var(
            model,
            torch.tensor(x_input, dtype=torch.float32, device=device)
        )

    pred_mu = float(mu.item())
    pred_var = float(var.item())

    ax.scatter(point[0, 0], point[0, 1], s=80)

    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.text(
        point[0, 0],
        point[0, 1],
        (
            f"μGT: {gt:.6f}\n"
            f"μ: {pred_mu:.6f}\n"
            f"σ²GT: {GT_VAR:.6f}\n"
            f"σ²: {pred_var:.6f}"
        ),
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.36, 0.36)
    ax.grid(True)

plt.savefig("stochastic_plots/five_random_pose_single_point.png", dpi=200)


# -------------------------------------------------
# 5 TRAINING SAMPLES FROM DATASET
# -------------------------------------------------
data = np.load(DATA_PATH)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax in axes:

    idx = np.random.randint(0, len(data))

    robot_xy = data[idx, 0:2]
    point = data[idx, 2:4]
    gt_mu = data[idx, 5]
    gt_var = data[idx, 6]

    x_input = data[idx, 0:4][None, :]

    with torch.no_grad():
        mu, var = predict_mu_var(
            model,
            torch.tensor(x_input, dtype=torch.float32, device=device)
        )

    pred_mu = float(mu.item())
    pred_var = float(var.item())

    ax.scatter(point[0], point[1], s=80)

    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.text(
        point[0],
        point[1],
        (
            f"μGT: {gt_mu:.6f}\n"
            f"μ: {pred_mu:.6f}\n"
            f"σ²GT: {gt_var:.6f}\n"
            f"σ²: {pred_var:.6f}"
        ),
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.36, 0.36)
    ax.grid(True)

plt.savefig("stochastic_plots/five_training_pose_single_point_dataset.png", dpi=200)

print("Done.")