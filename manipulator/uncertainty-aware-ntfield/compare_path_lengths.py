import numpy as np
from scipy.stats import wilcoxon

DIRS = {
    '0cm': 'Evaluations/Arm/paths_0cm',
    '1cm': 'Evaluations/Arm/paths_1cm',
    '2cm': 'Evaluations/Arm/paths_2cm',
}

# ── Load clean indices and lengths for each pipeline ──────────────────────────
data = {}
for label, d in DIRS.items():
    indices    = np.load(f'{d}/clean_pair_indices.npy')
    ee_lengths = np.load(f'{d}/clean_ee_lengths.npy')
    cs_lengths = np.load(f'{d}/clean_cs_lengths.npy')
    # store as dict: pair_idx -> (ee, cs)
    data[label] = {int(idx): (ee, cs)
                   for idx, ee, cs in zip(indices, ee_lengths, cs_lengths)}
    print(f'{label}: {len(indices)} clean trajectories  '
          f'(indices: {sorted(indices.tolist())})')

# ── Intersection of clean pair indices across all three pipelines ──────────────
shared = set(data['0cm'].keys())
for label in ('1cm', '2cm'):
    shared &= set(data[label].keys())
shared = sorted(shared)
print(f'\nShared clean pairs across all pipelines: {len(shared)}')
print(f'Indices: {shared}')

# ── Build aligned arrays over shared pairs ────────────────────────────────────
ee = {label: np.array([data[label][i][0] for i in shared]) for label in DIRS}
cs = {label: np.array([data[label][i][1] for i in shared]) for label in DIRS}

# ── Normalise by 0cm baseline (per-pair ratio) ────────────────────────────────
print('\n── EE path length (ratio vs 0cm) ─────────────────────────────────────')
print(f'  {"Pipeline":<8}  {"mean ratio":>12}  {"std":>8}  {"median":>8}  {"p-value vs 0cm":>16}')
for label in ('0cm', '1cm', '2cm'):
    ratio = ee[label] / ee['0cm']
    if label == '0cm':
        print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  {np.median(ratio):>8.4f}  {"(baseline)":>16}')
    else:
        _, p = wilcoxon(ee['0cm'], ee[label])
        print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  {np.median(ratio):>8.4f}  {p:>16.4e}')

print('\n── C-space path length (ratio vs 0cm) ────────────────────────────────')
print(f'  {"Pipeline":<8}  {"mean ratio":>12}  {"std":>8}  {"median":>8}  {"p-value vs 0cm":>16}')
for label in ('0cm', '1cm', '2cm'):
    ratio = cs[label] / cs['0cm']
    if label == '0cm':
        print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  {np.median(ratio):>8.4f}  {"(baseline)":>16}')
    else:
        _, p = wilcoxon(cs['0cm'], cs[label])
        print(f'  {label:<8}  {ratio.mean():>12.4f}  {ratio.std():>8.4f}  {np.median(ratio):>8.4f}  {p:>16.4e}')

# ── Raw lengths for reference ──────────────────────────────────────────────────
print('\n── Raw EE lengths (m) over shared pairs ──────────────────────────────')
for label in DIRS:
    print(f'  {label:<8}  mean={ee[label].mean():.4f}  std={ee[label].std():.4f}  median={np.median(ee[label]):.4f}')

print('\n── Raw C-space lengths (rad) over shared pairs ───────────────────────')
for label in DIRS:
    print(f'  {label:<8}  mean={cs[label].mean():.4f}  std={cs[label].std():.4f}  median={np.median(cs[label]):.4f}')

# ── Save aligned arrays for downstream use ────────────────────────────────────
np.save('Evaluations/Arm/shared_pair_indices.npy', np.array(shared))
for label in DIRS:
    np.save(f'Evaluations/Arm/shared_ee_{label}.npy', ee[label])
    np.save(f'Evaluations/Arm/shared_cs_{label}.npy', cs[label])
print('\nSaved shared arrays to Evaluations/Arm/')