import numpy as np
from scipy.stats import wilcoxon
import torch
import pytorch_kinematics as pk
from pathlib import Path
DIRS = {
    '0cm':  'Evaluations/Arm/paths_0cm',
    # '0.6cm': 'Evaluations/Arm/paths_0.6cm',
    '1cm':  'Evaluations/Arm/paths_1cm',
    '2cm':  'Evaluations/Arm/paths_2cm',
    'ours': 'Evaluations/Arm/paths_ours',
}

SCALE = np.pi / 0.5

# ── Build FK chain once ────────────────────────────────────────────────────────
def build_chain():
    chain = pk.build_serial_chain_from_urdf(
        open('datasets/arm/UR5/ur5e.urdf').read(), 'wrist_3_link')
    return chain.to(dtype=torch.float32, device='cuda')

def cs_length(path_rad):
    diffs = np.diff(path_rad, axis=0)
    return np.linalg.norm(diffs, axis=1).sum()

def ee_length(path_rad, chain):
    q  = torch.tensor(path_rad, dtype=torch.float32, device='cuda')
    tg = chain.forward_kinematics(q, end_only=True)
    ee = tg.get_matrix()[:, :3, 3]              # (T, 3)
    return (ee[1:] - ee[:-1]).norm(dim=1).sum().item()

# ── Load successful indices and compute both lengths ──────────────────────────
chain = build_chain()
data  = {}
for label, d in DIRS.items():
    successes = np.load(f'{d}/successes.npy')
    indices   = np.where(successes)[0]
    indices   = [i for i in indices
                 if not (Path(d) / f'path_{i:03d}_collision.npy').exists()]
    lengths   = {}
    for i in indices:
        path_rad = np.load(f'{d}/path_{i:03d}.npy')
        lengths[int(i)] = {
            'cs': cs_length(path_rad),
            'ee': ee_length(path_rad, chain),
        }
    data[label] = lengths
    print(f'{label}: {len(indices)} successful trajectories')

# ── Intersection ───────────────────────────────────────────────────────────────
labels = list(DIRS)
shared = set(data[labels[0]].keys())
for label in labels[1:]:
    shared &= set(data[label].keys())
shared = sorted(shared)
print(f'\nShared successful pairs: {len(shared)}')

# ── Build aligned arrays ───────────────────────────────────────────────────────
cs = {label: np.array([data[label][i]['cs'] for i in shared]) for label in DIRS}
ee = {label: np.array([data[label][i]['ee'] for i in shared]) for label in DIRS}

# ── Print results ──────────────────────────────────────────────────────────────
BASELINE = '0cm'

for metric_name, metric in [('C-space (rad)', cs), ('EE (m)', ee)]:
    print(f'\n── {metric_name} path length (ratio vs {BASELINE}) ──────────────────────')
    print(f'  {"Pipeline":<8}  {"mean ratio":>12}  {"std":>8}  {"median":>8}  {"p-value vs " + BASELINE:>16}')
    for label in DIRS:
        ratio = metric[label] / metric[BASELINE]
        if label == BASELINE:
            print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  '
                  f'{np.median(ratio):>8.4f}  {"(baseline)":>16}')
        else:
            _, p = wilcoxon(metric[BASELINE], metric[label])
            print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  '
                  f'{np.median(ratio):>8.4f}  {p:>16.4e}')

    print(f'\n── Raw {metric_name} lengths over shared pairs ───────────────────')
    for label in DIRS:
        print(f'  {label:<8}  mean={metric[label].mean():.4f}  '
              f'std={metric[label].std():.4f}  median={np.median(metric[label]):.4f}')