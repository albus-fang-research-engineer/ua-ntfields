"""
Compare a folder of optimized paths (path_000.npy … path_099.npy) against
the 0cm baseline.

Lengths are computed from raw path files (identical to the reference script):
  - C-space : sum of per-step Euclidean norms in joint space (rad)
  - EE       : sum of per-step end-effector displacements via FK (m)

Success cases are assumed to be the same as 0cm (successes.npy).
"""

import sys
import numpy as np
import torch
import pytorch_kinematics as pk
from scipy.stats import wilcoxon
from pathlib import Path

sys.path.append('.')

# ── Config ────────────────────────────────────────────────────────────────────
DIR_0CM   = Path("Evaluations/Arm/paths_0cm")
NEW_DIR   = Path("Evaluations/Arm/optimized_paths_speed_up_tuning")   # ← change if needed
URDF_PATH = "datasets/arm/UR5/ur5e.urdf"
END_LINK  = "wrist_3_link"
LABEL_NEW = "optimized"
# NOMINAL_DIR =Path("Evaluations/Arm/paths_ours")
# ── FK chain ──────────────────────────────────────────────────────────────────
def build_chain():
    chain = pk.build_serial_chain_from_urdf(
        open(URDF_PATH).read(), END_LINK)
    return chain.to(dtype=torch.float32, device='cuda')

# ── Length functions (identical to reference script) ──────────────────────────
def cs_length(path_rad):
    diffs = np.diff(path_rad, axis=0)
    return np.linalg.norm(diffs, axis=1).sum()

def ee_length(path_rad, chain):
    q  = torch.tensor(path_rad, dtype=torch.float32, device='cuda')
    tg = chain.forward_kinematics(q, end_only=True)
    ee = tg.get_matrix()[:, :3, 3]          # (T, 3)
    return (ee[1:] - ee[:-1]).norm(dim=1).sum().item()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    chain = build_chain()

    # Shared indices come from 0cm successes (assumed same for new folder)
    successes = np.load(DIR_0CM / "successes.npy")
    shared    = sorted(np.where(successes)[0].tolist())
    print(f"0cm successful trajectories : {len(shared)}")

    # Keep only paths with no collision in 0cm (no path_***_collision.npy file)
    shared = [i for i in shared if not (DIR_0CM / f"path_{i:03d}_collision.npy").exists()]
    print(f"0cm collision-free trajectories : {len(shared)}")
    print(f"Indices: {shared}\n")
    # shared = [i for i in shared if not (NOMINAL_DIR / f"path_{i:03d}_collision.npy").exists()]
    data = {'0cm': {}, LABEL_NEW: {}}

    for i in shared:
        # 0cm baseline
        path_0cm = np.load(DIR_0CM / f"path_{i:03d}.npy")
        data['0cm'][i] = {
            'cs': cs_length(path_0cm),
            'ee': ee_length(path_0cm, chain),
        }

        # New pipeline
        p = NEW_DIR / f"path_{i:03d}.npy"
        if not p.exists():
            print(f"  WARNING: {p} missing — skipping index {i}")
            continue
        path_new = np.asarray(np.load(p, allow_pickle=True))
        data[LABEL_NEW][i] = {
            'cs': cs_length(path_new),
            'ee': ee_length(path_new, chain),
        }

    # Drop any index missing from new folder
    shared = sorted(set(data['0cm'].keys()) & set(data[LABEL_NEW].keys()))
    print(f"Pairs used for comparison : {len(shared)}\n")

    cs = {label: np.array([data[label][i]['cs'] for i in shared]) for label in ('0cm', LABEL_NEW)}
    ee = {label: np.array([data[label][i]['ee'] for i in shared]) for label in ('0cm', LABEL_NEW)}

    # ── Per-pair ratios ───────────────────────────────────────────────────────
    print('\n── Per-pair ratios (optimized / 0cm) ─────────────────────────────────')
    print(f'  {"idx":>4}  {"CS ratio":>10}  {"EE ratio":>10}')
    for idx, cs_r, ee_r in zip(shared, cs[LABEL_NEW] / cs['0cm'], ee[LABEL_NEW] / ee['0cm']):
        print(f'  {idx:>4}  {cs_r:>10.4f}  {ee_r:>10.4f}')

    # ── Print results (same format as reference script) ───────────────────────
    for metric_name, metric in [('C-space (rad)', cs), ('EE (m)', ee)]:
        print(f'\n── {metric_name} path length (ratio vs 0cm) ──────────────────────')
        print(f'  {"Pipeline":<12}  {"mean ratio":>12}  {"std":>8}  {"median":>8}  {"p-value vs 0cm":>16}')

        for label in ('0cm', LABEL_NEW):
            ratio = metric[label] / metric['0cm']
            if label == '0cm':
                print(f'  {label:<12}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  '
                      f'{np.median(ratio):>8.4f}  {"(baseline)":>16}')
            else:
                _, p = wilcoxon(metric['0cm'], metric[label])
                print(f'  {label:<12}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  '
                      f'{np.median(ratio):>8.4f}  {p:>16.4e}')

        print(f'\n── Raw {metric_name} lengths over shared pairs ───────────────────')
        for label in ('0cm', LABEL_NEW):
            print(f'  {label:<12}  mean={metric[label].mean():.4f}  '
                  f'std={metric[label].std():.4f}  median={np.median(metric[label]):.4f}')


if __name__ == "__main__":
    main()