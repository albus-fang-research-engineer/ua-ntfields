"""
serial_chain_fk.py

Hand-rolled forward kinematics for a serial revolute chain (e.g. UR5e),
designed as a drop-in replacement for pytorch_kinematics inside
MeshDistanceModel.

Per call: ~25 elementary torch ops instead of ~400 with pytorch_kinematics.
URDF parsing happens once at construction; runtime sees only constants.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch


# ─── URDF parsing ───────────────────────────────────────────────────────────

def _rpy_xyz_to_matrix(rpy, xyz):
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = xyz
    return T


def _parse_joint(j_elem):
    """Return (T_origin 4x4, axis 3-vec, jtype) for a URDF <joint> element."""
    jtype = j_elem.attrib["type"]

    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    origin = j_elem.find("origin")
    if origin is not None:
        if "xyz" in origin.attrib:
            xyz = list(map(float, origin.attrib["xyz"].split()))
        if "rpy" in origin.attrib:
            rpy = list(map(float, origin.attrib["rpy"].split()))
    T_origin = _rpy_xyz_to_matrix(rpy, xyz)

    axis = np.array([0.0, 0.0, 1.0])
    axis_elem = j_elem.find("axis")
    if axis_elem is not None and "xyz" in axis_elem.attrib:
        axis = np.array(list(map(float, axis_elem.attrib["xyz"].split())))
        n = np.linalg.norm(axis)
        if n > 1e-12:
            axis = axis / n

    return T_origin, axis, jtype


def extract_serial_chain(urdf_path, link_names):
    """
    Walk a URDF from the root toward each named link in order, collapsing any
    fixed joints into the next revolute joint's origin. Returns per-link
    constants suitable for a serial revolute FK.

    Parameters
    ----------
    urdf_path  : str
    link_names : list[str]    ordered chain of LINK names (base outward)

    Returns
    -------
    origins : (n, 4, 4)  T_origin for each link (incl. any fixed-joint prefix)
    axes    : (n, 3)     unit revolute axis (in joint frame) for each link
    """
    root = ET.parse(urdf_path).getroot()

    joints_by_child = {}
    for j in root.findall("joint"):
        joints_by_child[j.find("child").attrib["link"]] = j

    origins, axes = [], []
    for i, link_name in enumerate(link_names):
        prev_link = link_names[i - 1] if i > 0 else None

        # Walk backward from this link to either prev_link or the URDF root.
        chain = []
        cursor = link_name
        while cursor in joints_by_child:
            j = joints_by_child[cursor]
            chain.append(j)
            cursor = j.find("parent").attrib["link"]
            if cursor == prev_link:
                break

        # Compose joints in parent->child order; only ONE revolute is allowed
        # in each segment (otherwise the serial assumption is violated).
        T = np.eye(4)
        revolute_axis = None
        for j in reversed(chain):
            T_o, ax, jt = _parse_joint(j)
            T = T @ T_o
            if jt in ("revolute", "continuous"):
                if revolute_axis is not None:
                    raise ValueError(
                        f"Multiple revolute joints between {prev_link} and "
                        f"{link_name}; chain isn't serial."
                    )
                revolute_axis = ax

        if revolute_axis is None:
            raise ValueError(f"No revolute joint leading to {link_name}.")

        origins.append(T)
        axes.append(revolute_axis)

    return np.stack(origins), np.stack(axes)


# ─── Torch FK ───────────────────────────────────────────────────────────────

class SerialChainFK:
    """
    Forward kinematics for a serial chain of revolute joints.

    Call signature:
        fk(q)        # q : (B, n) or (n,)  ->  returns (B, n, 4, 4)

    The returned tensor matches link_names: out[:, i] is the world-frame
    homogeneous transform of link_names[i].
    """

    def __init__(self, urdf_path, link_names, device="cuda", dtype=torch.float32):
        origins_np, axes_np = extract_serial_chain(urdf_path, link_names)

        self.device     = device
        self.dtype      = dtype
        self.n_links    = len(link_names)
        self.link_names = list(link_names)

        self.origins = torch.as_tensor(origins_np, dtype=dtype, device=device)  # (n,4,4)
        self.axes    = torch.as_tensor(axes_np,    dtype=dtype, device=device)  # (n,3)

        # Specialization: if every axis is +z, skip Rodrigues entirely.
        self._all_z = bool(np.allclose(axes_np, np.array([[0.0, 0.0, 1.0]] * self.n_links)))

    def __call__(self, q):
        if q.dim() == 1:
            q = q.unsqueeze(0)
        B, n = q.shape
        assert n == self.n_links, f"expected {self.n_links} joints, got {n}"

        c = torch.cos(q)  # (B, n)
        s = torch.sin(q)  # (B, n)

        T_local = self._build_local_z(c, s) if self._all_z \
                  else self._build_local_general(c, s)

        # Serial chain composition. 6 sequential matmuls; can't be parallelized
        # across joints, but each one is a single op so dispatch cost is tiny.
        T_world = [T_local[:, 0]]
        T_curr = T_world[0]
        for i in range(1, n):
            T_curr = T_curr @ T_local[:, i]
            T_world.append(T_curr)
        return torch.stack(T_world, dim=1)  # (B, n, 4, 4)

    # ---- specialized z-axis path (UR5e and most 6-DoF arms) ----------------

    def _build_local_z(self, c, s):
        """
        T_local = T_origin @ Rz(q). Rz only rotates the first two columns,
        so we fold the rotation directly into T_origin's columns and skip
        constructing an intermediate Rz matrix.
        """
        B = c.shape[0]
        O = self.origins                                  # (n, 4, 4)

        col0_o = O[:, :, 0].unsqueeze(0)                  # (1, n, 4)
        col1_o = O[:, :, 1].unsqueeze(0)                  # (1, n, 4)
        col2_o = O[:, :, 2].unsqueeze(0).expand(B, -1, -1)
        col3_o = O[:, :, 3].unsqueeze(0).expand(B, -1, -1)

        cb = c.unsqueeze(-1)                              # (B, n, 1)
        sb = s.unsqueeze(-1)

        col0 =  col0_o * cb + col1_o * sb                 # (B, n, 4)
        col1 = -col0_o * sb + col1_o * cb

        return torch.stack([col0, col1, col2_o, col3_o], dim=-1)  # (B, n, 4, 4)

    # ---- general axis-angle path (Rodrigues) -------------------------------

    def _build_local_general(self, c, s):
        """
        R(axis, q) by Rodrigues. Builds the 9 entries directly to avoid
        in-place ops; autograd-friendly.
        """
        B, n = c.shape
        kx = self.axes[:, 0]                              # (n,)
        ky = self.axes[:, 1]
        kz = self.axes[:, 2]
        omc = 1.0 - c                                     # (B, n)

        # Entries of R, each (B, n).
        R00 = c   + omc * kx * kx
        R01 = omc * kx * ky - s * kz
        R02 = omc * kx * kz + s * ky
        R10 = omc * ky * kx + s * kz
        R11 = c   + omc * ky * ky
        R12 = omc * ky * kz - s * kx
        R20 = omc * kz * kx - s * ky
        R21 = omc * kz * ky + s * kx
        R22 = c   + omc * kz * kz

        row0 = torch.stack([R00, R01, R02], dim=-1)       # (B, n, 3)
        row1 = torch.stack([R10, R11, R12], dim=-1)
        row2 = torch.stack([R20, R21, R22], dim=-1)
        R3   = torch.stack([row0, row1, row2], dim=-2)    # (B, n, 3, 3)

        # T_local = T_origin @ [[R3 0]; [0 1]] simplifies to composing only
        # the rotation block of T_origin and re-using its translation.
        Ro = self.origins[:, :3, :3].unsqueeze(0)         # (1, n, 3, 3)
        to = self.origins[:, :3, 3].unsqueeze(0)          # (1, n, 3)
        Rc = Ro @ R3                                      # (B, n, 3, 3)

        # Assemble (B, n, 4, 4) without in-place writes.
        to_b   = to.unsqueeze(-1).expand(B, n, 3, 1)      # (B, n, 3, 1)
        top    = torch.cat([Rc, to_b], dim=-1)            # (B, n, 3, 4)
        bottom = torch.zeros(B, n, 1, 4,
                             device=self.device, dtype=self.dtype)
        bottom = bottom + torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype
        ).view(1, 1, 1, 4)
        return torch.cat([top, bottom], dim=-2)           # (B, n, 4, 4)


# ─── Verification helper ────────────────────────────────────────────────────

def max_diff_vs_pytorch_kinematics(
    fk, urdf_path, end_link, link_names, q, device="cuda", dtype=torch.float32,
):
    """
    Returns the maximum absolute difference between this FK and
    pytorch_kinematics on a given joint config. Should be < 1e-5 in float32.
    """
    import pytorch_kinematics as pk

    with open(urdf_path) as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), end_link)
    chain = chain.to(dtype=dtype, device=device)

    q_t = torch.as_tensor(q, dtype=dtype, device=device).unsqueeze(0)

    pk_out  = chain.forward_kinematics(q_t, end_only=False)
    pk_mats = torch.stack([pk_out[name].get_matrix()[0] for name in link_names])
    our_mats = fk(q_t)[0]

    return (pk_mats - our_mats).abs().max().item()


# ─── Self-test: run `python serial_chain_fk.py` ─────────────────────────────

if __name__ == "__main__":
    import os
    import time

    # ----- edit these for your setup ---------------------------------------
    URDF_PATH = "/workspace/datasets/arm/UR5/ur5e.urdf"
    LINK_NAMES = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    END_LINK = "wrist_3_link"
    # -----------------------------------------------------------------------

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.float32
    N_BENCH = 200

    if not os.path.exists(URDF_PATH):
        raise SystemExit(f"URDF not found: {URDF_PATH}  -- edit URDF_PATH at top of __main__")

    print(f"Device : {DEVICE}")
    print(f"Dtype  : {DTYPE}")
    print(f"URDF   : {URDF_PATH}")
    print()

    # ---- Build our FK ------------------------------------------------------
    fk = SerialChainFK(URDF_PATH, LINK_NAMES, device=DEVICE, dtype=DTYPE)
    print(f"Joint axes parsed from URDF:")
    for name, ax in zip(LINK_NAMES, fk.axes.cpu().numpy()):
        print(f"  {name:20s}  axis = {ax}")
    print(f"Using all-z specialization: {fk._all_z}")
    print()

    # ---- Build pytorch_kinematics reference --------------------------------
    try:
        import pytorch_kinematics as pk
    except ImportError:
        raise SystemExit("pytorch_kinematics not installed; cannot verify.")

    with open(URDF_PATH) as f:
        pk_chain = pk.build_serial_chain_from_urdf(f.read(), END_LINK)
    pk_chain = pk_chain.to(dtype=DTYPE, device=DEVICE)

    def fk_pk(q_t):
        """pk wrapper that returns the same (B, n, 4, 4) shape as ours."""
        out = pk_chain.forward_kinematics(q_t, end_only=False)
        mats = [out[name].get_matrix() for name in LINK_NAMES]   # each (B, 4, 4)
        return torch.stack(mats, dim=1)                          # (B, n, 4, 4)

    # ====================================================================
    # CORRECTNESS
    # ====================================================================
    print("=" * 60)
    print("CORRECTNESS  (max |Δ| over all link transforms)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    test_qs = [
        ("zeros",        np.zeros(6)),
        ("small",        np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])),
        ("mid",          np.array([0.5, -0.5, 0.3, -0.7, 0.2, 0.1])),
        ("random 1",     rng.uniform(-np.pi, np.pi, 6)),
        ("random 2",     rng.uniform(-np.pi, np.pi, 6)),
        ("random 3",     rng.uniform(-np.pi, np.pi, 6)),
        ("near limits",  np.array([np.pi - 0.01, -np.pi + 0.01, 0.0, np.pi/2, -np.pi/2, np.pi])),
    ]
    THRESH = 1e-4
    worst = 0.0
    for label, q in test_qs:
        err = max_diff_vs_pytorch_kinematics(
            fk, URDF_PATH, END_LINK, LINK_NAMES, q, device=DEVICE, dtype=DTYPE,
        )
        status = "OK" if err < THRESH else "FAIL"
        print(f"  {label:14s}  max |Δ| = {err:.2e}   [{status}]")
        worst = max(worst, err)
    print(f"\n  worst case: {worst:.2e}   threshold: {THRESH:.0e}")

    # Per-link breakdown on one random config
    print("\n  --- per-link diff on a random q -----------------")
    q_t = torch.as_tensor(rng.uniform(-np.pi, np.pi, 6), dtype=DTYPE, device=DEVICE).unsqueeze(0)
    pk_mats  = fk_pk(q_t)[0]    # (n, 4, 4)
    our_mats = fk(q_t)[0]
    for i, name in enumerate(LINK_NAMES):
        d = (pk_mats[i] - our_mats[i]).abs().max().item()
        print(f"  {name:20s}  max |Δ| = {d:.2e}")

    if worst >= THRESH:
        print("\n  *** correctness FAILED -- not safe to swap in yet ***")
        raise SystemExit(1)

    # ====================================================================
    # FORWARD TIMING
    # ====================================================================
    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    print("\n" + "=" * 60)
    print(f"FORWARD TIMING  ({N_BENCH} iters, CUDA-synced)")
    print("=" * 60)
    q_t = torch.as_tensor(rng.uniform(-np.pi, np.pi, 6), dtype=DTYPE, device=DEVICE).unsqueeze(0)

    # warmup
    for _ in range(20):
        _ = fk(q_t); _ = fk_pk(q_t)
    sync()

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        _ = fk(q_t)
    sync()
    t_fwd_ours = (time.perf_counter() - t0) / N_BENCH

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        _ = fk_pk(q_t)
    sync()
    t_fwd_pk = (time.perf_counter() - t0) / N_BENCH

    print(f"  ours                : {t_fwd_ours * 1e3:7.3f} ms/call")
    print(f"  pytorch_kinematics  : {t_fwd_pk   * 1e3:7.3f} ms/call")
    print(f"  speedup             : {t_fwd_pk / t_fwd_ours:5.1f}x")

    # ====================================================================
    # FORWARD + BATCHED-BACKWARD TIMING  (mirrors mu_sigma_grad_mesh)
    # ====================================================================
    print("\n" + "=" * 60)
    print(f"FORWARD + BATCHED BACKWARD  (K=60 outputs, mirrors query())")
    print("=" * 60)
    K = 60  # representative # of "near" constraints

    def bench_grad(fk_fn):
        # warmup
        for _ in range(10):
            q = torch.as_tensor(rng.uniform(-np.pi, np.pi, 6),
                                dtype=DTYPE, device=DEVICE
                                ).unsqueeze(0).requires_grad_(True)
            T_w = fk_fn(q)                       # (1, n, 4, 4)
            y   = T_w.reshape(-1)[:K]            # (K,)
            I_K = torch.eye(K, device=DEVICE, dtype=DTYPE)
            _ = torch.autograd.grad(y, q, grad_outputs=I_K, is_grads_batched=True)
        sync()

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            q = torch.as_tensor(rng.uniform(-np.pi, np.pi, 6),
                                dtype=DTYPE, device=DEVICE
                                ).unsqueeze(0).requires_grad_(True)
            T_w = fk_fn(q)
            y   = T_w.reshape(-1)[:K]
            I_K = torch.eye(K, device=DEVICE, dtype=DTYPE)
            _ = torch.autograd.grad(y, q, grad_outputs=I_K, is_grads_batched=True)
        sync()
        return (time.perf_counter() - t0) / N_BENCH

    t_grad_ours = bench_grad(fk)
    t_grad_pk   = bench_grad(fk_pk)

    print(f"  ours                : {t_grad_ours * 1e3:7.3f} ms/call")
    print(f"  pytorch_kinematics  : {t_grad_pk   * 1e3:7.3f} ms/call")
    print(f"  speedup             : {t_grad_pk / t_grad_ours:5.1f}x")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)