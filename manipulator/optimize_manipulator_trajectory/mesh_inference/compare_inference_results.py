"""
sanity_check_mesh_inference.py

Compare outputs of mesh_inference.py (autograd) against
mesh_inference_default.py (finite differences) on randomly sampled
joint configurations.

Checks per trial:
  - mu       (signed distances)                 - tight tolerance
  - sigma    (constant)                         - exact equality
  - obs_k    (selected obstacle indices)        - exact equality
  - grad     (joint-space Jacobian)             - finite-difference tolerance

Run:
    python sanity_check_mesh_inference.py
"""

import numpy as np
import torch

# Both implementations side by side
from mesh_inference         import MeshDistanceModel as MeshNew
from mesh_inference_default import MeshDistanceModel as MeshOld


# ── Configure these paths to match your setup ─────────────────────────────
URDF_PATH  = "/workspace/datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "/workspace/datasets/arm/UR5/meshes/sphere/sphere"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Test settings ─────────────────────────────────────────────────────────
N_TRIALS    = 10
N_OBSTACLES = 5000
Q_RANGE     = (-np.pi / 2, np.pi / 2)   # joint sampling range [rad]
SEED        = 42

# Tolerances
ATOL_MU      = 1e-5      # forward computations are essentially identical
ATOL_GRAD    = 1e-3      # autograd vs central-diff at eps = 1e-4
RTOL_GRAD    = 1e-2      # for rows with large gradient magnitudes


def make_obstacle_cloud(n_obs, device, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, size=(n_obs, 3))
    return torch.tensor(pts, dtype=torch.float32, device=device)


def compare(mu_o, sg_o, gr_o, ok_o,
            mu_n, sg_n, gr_n, ok_n):
    """Compute per-quantity stats and pass/fail flags."""
    report = {}

    # --- mu ---
    mu_diff = np.abs(mu_o - mu_n)
    report["mu_max_abs"] = float(mu_diff.max())
    report["mu_pass"]    = bool(np.allclose(mu_o, mu_n, atol=ATOL_MU))

    # --- sigma ---
    report["sigma_pass"] = bool(np.array_equal(sg_o, sg_n))

    # --- obs_k ---
    report["obs_k_pass"]      = bool(np.array_equal(ok_o, ok_n))
    report["obs_k_n_mismatch"] = int(np.sum(ok_o != ok_n))

    # --- grad ---
    # Compare only entries where the old code populated a gradient
    # (i.e. where mu < GRAD_THRESHOLD). Far-field entries are zeros in both.
    grad_diff = np.abs(gr_o - gr_n)
    report["grad_max_abs"] = float(grad_diff.max())

    # Per-row relative error on the rows with non-trivial gradient magnitude
    row_norms_old = np.linalg.norm(gr_o, axis=1)
    nontrivial    = row_norms_old > 1e-6
    if nontrivial.any():
        row_diff   = np.linalg.norm(gr_o[nontrivial] - gr_n[nontrivial], axis=1)
        row_rel    = row_diff / row_norms_old[nontrivial]
        report["grad_rel_max_per_row"]  = float(row_rel.max())
        report["grad_rel_mean_per_row"] = float(row_rel.mean())
        report["grad_pass"] = bool(
            (row_rel.max() < RTOL_GRAD) or (grad_diff.max() < ATOL_GRAD)
        )
    else:
        report["grad_rel_max_per_row"]  = 0.0
        report["grad_rel_mean_per_row"] = 0.0
        report["grad_pass"] = bool(grad_diff.max() < ATOL_GRAD)

    passed = all(
        report[k] for k in ("mu_pass", "sigma_pass", "obs_k_pass", "grad_pass")
    )
    return passed, report


def print_trial(trial, q, passed, report):
    status = "PASS" if passed else "FAIL"
    print(f"Trial {trial:2d}: {status}")
    print(f"   q                  : {np.array2string(q, precision=3)}")
    print(f"   mu  max |Δ|        : {report['mu_max_abs']:.3e}     "
          f"{'OK' if report['mu_pass']    else 'FAIL'}")
    print(f"   sigma              : "
          f"{'OK' if report['sigma_pass'] else 'FAIL'}")
    print(f"   obs_k mismatches   : {report['obs_k_n_mismatch']:3d}        "
          f"{'OK' if report['obs_k_pass'] else 'FAIL'}")
    print(f"   grad max |Δ|       : {report['grad_max_abs']:.3e}     "
          f"{'OK' if report['grad_pass']  else 'FAIL'}")
    print(f"   grad per-row rel   : "
          f"max={report['grad_rel_max_per_row']:.3e}  "
          f"mean={report['grad_rel_mean_per_row']:.3e}")
    print()


def main():
    print("=" * 64)
    print(" Sanity check: mesh_inference.py  vs  mesh_inference_default.py")
    print("=" * 64)
    print(f" Device   : {DEVICE}")
    print(f" Trials   : {N_TRIALS}")
    print(f" Obstacles: {N_OBSTACLES}")
    print()

    obs_pts = make_obstacle_cloud(N_OBSTACLES, DEVICE, seed=SEED)

    print("Building MeshDistanceModel (OLD: finite differences) ...")
    model_old = MeshOld(
        urdf_path       = URDF_PATH,
        sphere_dir      = SPHERE_DIR,
        obstacle_points = obs_pts,
        device          = DEVICE,
    )

    print("Building MeshDistanceModel (NEW: autograd) ...")
    model_new = MeshNew(
        urdf_path       = URDF_PATH,
        sphere_dir      = SPHERE_DIR,
        obstacle_points = obs_pts,
        device          = DEVICE,
    )
    print()

    rng = np.random.default_rng(SEED + 1)
    all_passed = True

    for trial in range(N_TRIALS):
        q = rng.uniform(Q_RANGE[0], Q_RANGE[1], size=6).astype(np.float64)

        mu_o, sg_o, gr_o, ok_o = model_old.query(q)
        mu_n, sg_n, gr_n, ok_n = model_new.query(q)

        passed, report = compare(mu_o, sg_o, gr_o, ok_o,
                                 mu_n, sg_n, gr_n, ok_n)
        print_trial(trial, q, passed, report)

        if not passed:
            all_passed = False

    print("=" * 64)
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 64)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())