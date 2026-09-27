import numpy as np

def npy_to_off(npy_path, off_path):
    v = np.load(npy_path)
    with open(off_path, 'w') as f:
        f.write('OFF\n')
        f.write(f'{len(v)} 0 0\n')
        for vert in v:
            f.write(f'{vert[0]} {vert[1]} {vert[2]}\n')

npy_to_off('occupied_voxel_centers.npy', 'occupied_voxel_centers.off')
