"""
verify_mesh_inference_swap.py

Numerical regression test for the SerialChainFK swap inside
mesh_inference.py. Matches the construction in optimize_all_paths.py.

Workflow:
    1. With the OLD mesh_inference.py (pytorch_kinematics) still in place:
           python verify_mesh_inference_swap.py
       This saves reference outputs to mesh_inference_ref.npz.

    2. Swap in the new mesh_inference.py.

    3. Run again:
           python verify_mesh_inference_swap.py
       This loads the reference and compares the new outputs to it.

The reference file is auto-detected: missing -> save mode, present -> compare.
Delete it to regenerate.
"""

import sys
from pathlib import Path

import igl
import numpy as np

# IMPORTANT: patch GRAD_THRESHOLD *before* importing MeshDistanceModel.
# Default is 0.10 m, which means random joint configs (mu min ~0.3 m typically)
# produce zero "near" constraints and the autograd path never runs. We push it
# to 100 m so every constraint produces a real gradient — that way the test
# actually exercises the vmap'd backward we care about.
import mesh_inference.mesh_inference as _mi_mod
_PATCHED_THRESHOLD = 100.0
_mi_mod.GRAD_THRESHOLD = _PATCHED_THRESHOLD

from mesh_inference.mesh_inference import MeshDistanceModel, mu_sigma_grad_mesh


# ---- match optimize_all_paths.py exactly ----------------------------------
OBS_PATH   = "realpc_scaled_downsampled.off"
URDF_PATH  = "datasets/arm/UR5/ur5e.urdf"
SPHERE_DIR = "datasets/arm/UR5/meshes/sphere/sphere"
DEVICE     = "cuda"

REF_PATH   = Path("mesh_inference_ref.npz")
N_TESTS    = 10
SEED       = 0

# Pass criteria. mu is in meters; gradients accumulate roundoff through 6
# chained matmuls and one sqrt, so a slightly looser threshold is reasonable.
MU_TOL     = 1e-4
GRAD_TOL   = 1e-3


def build_mesh_model() -> MeshDistanceModel:
    v, _ = igl.read_triangle_mesh(OBS_PATH)
    obs = v.astype(np.float32)
    print(f"Obstacle cloud shape: {obs.shape}")
    print(f"GRAD_THRESHOLD patched to {_PATCHED_THRESHOLD} (forces all "
          f"constraints to be 'near' so the autograd path is exercised)")
    model = MeshDistanceModel(
        urdf_path       = URDF_PATH,
        sphere_dir      = SPHERE_DIR,
        obstacle_points = obs,
        end_link        = "wrist_3_link",
        device          = DEVICE,
    )
    print("MeshDistanceModel built.\n")
    return model


def gen_qs() -> np.ndarray:
    """Deterministic random configs (so save and compare see the same qs)."""
    rng = np.random.default_rng(SEED)
    return rng.uniform(-np.pi, np.pi, size=(N_TESTS, 6))


def save_reference(mesh) -> None:
    qs = gen_qs()
    mus, grads, obs_ks = [], [], []
    print(f"[SAVE mode] no reference at {REF_PATH}; generating one.\n")
    for i, q in enumerate(qs):
        mu, sigma, grad, obs_k = mu_sigma_grad_mesh(q, mesh)
        mus.append(mu); grads.append(grad); obs_ks.append(obs_k)
        gn = np.linalg.norm(grad, axis=1)
        print(f"  q[{i:02d}]  mu min={mu.min(): .4f}  "
              f"|grad| max={gn.max():.3e}  "
              f"obs_k range=[{obs_k.min()},{obs_k.max()}]")
    np.savez(
        REF_PATH,
        qs     = qs,
        mus    = np.stack(mus),
        grads  = np.stack(grads),
        obs_ks = np.stack(obs_ks),
    )
    print(f"\nSaved reference for {N_TESTS} configs to {REF_PATH}.")
    print("Now swap mesh_inference.py to the new version and re-run this script.")


def compare_against_reference(mesh) -> int:
    ref = np.load(REF_PATH)
    qs  = ref["qs"]
    assert qs.shape == (N_TESTS, 6), f"unexpected ref shape: {qs.shape}"

    # Guard against a stale reference saved before GRAD_THRESHOLD was patched.
    if np.abs(ref["grads"]).max() < 1e-12:
        print(f"WARNING: reference at {REF_PATH} has all-zero gradients.")
        print("That means it was saved before GRAD_THRESHOLD was patched, so")
        print("the autograd path was never exercised. The current comparison")
        print("would be meaningless. Delete the file and redo SAVE + COMPARE:\n")
        print(f"    rm {REF_PATH}")
        print("    # (switch back to ORIGINAL mesh_inference.py)")
        print("    python verify_mesh_inference_swap.py    # SAVE")
        print("    # (switch to NEW mesh_inference.py)")
        print("    python verify_mesh_inference_swap.py    # COMPARE")
        return 1

    print(f"[COMPARE mode] reference loaded from {REF_PATH}.\n")
    print(f"  {'idx':>3}  {'Δmu':>10}  {'Δgrad':>10}  {'obs_k':>8}  status")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*8}  ------")

    worst_mu      = 0.0
    worst_grad    = 0.0
    obs_mismatches = 0
    hard_fails    = 0

    for i in range(N_TESTS):
        mu, _, grad, obs_k = mu_sigma_grad_mesh(qs[i], mesh)
        d_mu   = float(np.abs(mu   - ref["mus"][i]  ).max())
        d_grad = float(np.abs(grad - ref["grads"][i]).max())
        match  = bool(np.array_equal(obs_k, ref["obs_ks"][i]))

        worst_mu    = max(worst_mu,   d_mu)
        worst_grad  = max(worst_grad, d_grad)
        if not match:
            obs_mismatches += 1

        ok = d_mu < MU_TOL and d_grad < GRAD_TOL
        if not ok:
            hard_fails += 1
        flag = "OK" if (ok and match) else ("soft" if ok else "FAIL")
        print(f"  {i:>3d}  {d_mu:10.2e}  {d_grad:10.2e}  "
              f"{'match' if match else 'DIFF':>8}  {flag}")

    print()
    print(f"  worst Δmu   : {worst_mu:.2e}   (threshold {MU_TOL:.0e})")
    print(f"  worst Δgrad : {worst_grad:.2e}   (threshold {GRAD_TOL:.0e})")
    print(f"  obs_k mismatches: {obs_mismatches}/{N_TESTS}")
    print()

    if hard_fails:
        print(f"  *** {hard_fails}/{N_TESTS} hard failures — DO NOT trust the swap ***")
        return 1
    if obs_mismatches > N_TESTS // 5:
        print(f"  *** {obs_mismatches}/{N_TESTS} obs_k mismatches is more than expected; "
              f"investigate ***")
        return 1
    if obs_mismatches > 0:
        print(f"  {obs_mismatches} soft obs_k mismatch(es): likely just float32 roundoff "
              f"pushing a tied neighbor across a boundary. Acceptable.")
    print("  PASS — safe to use the new mesh_inference.py")
    return 0


def main() -> int:
    mesh = build_mesh_model()
    if not REF_PATH.exists():
        save_reference(mesh)
        return 0
    return compare_against_reference(mesh)


if __name__ == "__main__":
    sys.exit(main())