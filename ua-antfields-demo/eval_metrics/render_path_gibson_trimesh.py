import numpy as np
import trimesh
import os

# ==== CONFIG ====
MESH_PATH = "/antfields/data/mesh.obj"

NPY_OPT_PATH = "/antfields/Experiments/03_18_12_10/epoch_0950_optimized.npy"
NPY_NOM_PATH = "/antfields/Experiments/03_18_12_10/epoch_0950_nominal.npy"


def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Waypoint file not found: {path}")
    return np.load(path)


def ensure_3d(pts):
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    return pts


def main():
    mesh = trimesh.load(MESH_PATH)

    wp_opt = ensure_3d(load_waypoints(NPY_OPT_PATH))
    wp_nom = ensure_3d(load_waypoints(NPY_NOM_PATH))

    print(f"[INFO] Optimized shape: {wp_opt.shape}")
    print(f"[INFO] Nominal shape: {wp_nom.shape}")

    # 🔵 Optimized path
    opt_path = trimesh.load_path(wp_opt)
    opt_path.colors = np.tile([0, 0, 255, 255], (len(opt_path.entities), 1))

    # 🟠 Nominal path
    nom_path = trimesh.load_path(wp_nom)
    nom_path.colors = np.tile([255, 165, 0, 255], (len(nom_path.entities), 1))

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(opt_path)
    scene.add_geometry(nom_path)

    scene.show()


if __name__ == "__main__":
    main()