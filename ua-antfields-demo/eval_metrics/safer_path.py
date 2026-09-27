import numpy as np
import open3d as o3d
import os

# ==== CONFIG ====
MESH_PATH = "/antfields/data/mesh.obj"

NPY_OPT_PATH = "/antfields/Experiments/03_18_12_10/epoch_2100_optimized.npy"
NPY_NOM_PATH = "/antfields/Experiments/03_18_12_10/epoch_2100_nominal.npy"
NPY_NOM_PATH = "/antfields/baseline_path_collision_step3.npy"
NPY_OPT_PATH = "/antfields/Experiments/03_21_12_19/traj_200.npy"
def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Waypoint file not found: {path}")
    return np.load(path)


def ensure_3d(pts):
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    return pts


def create_lineset(points, color):
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

import numpy as np
import open3d as o3d

def create_tube(points, radius=0.03, color=[0, 0, 1]):
    meshes = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            continue

        direction /= length

        # Create cylinder along z-axis
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.paint_uniform_color(color)

        # Align cylinder with segment
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        if np.linalg.norm(v) > 1e-6:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
        else:
            R = np.eye(3)

        cylinder.rotate(R, center=np.zeros(3))

        # Move to midpoint
        midpoint = (p1 + p2) / 2
        cylinder.translate(midpoint)

        meshes.append(cylinder)
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.05)
        sphere.paint_uniform_color(color)
        sphere.translate(p)
        meshes.append(sphere)
    return meshes

def main():
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh.compute_vertex_normals()

    wp_opt = ensure_3d(load_waypoints(NPY_OPT_PATH))
    wp_nom = ensure_3d(load_waypoints(NPY_NOM_PATH))
    print(f"[INFO] Optimized shape: {wp_opt.shape}")
    print(f"[INFO] Nominal shape: {wp_nom.shape}")


    robot_pose = wp_nom[-4]

    black_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0105)
    black_sphere.paint_uniform_color([0, 0, 0])
    black_sphere.translate(robot_pose)
    # 🔵 Optimized
    # opt_lines = create_lineset(wp_opt, [0, 0, 1])

    # # 🟠 Nominal
    # nom_lines = create_lineset(wp_nom, [1, 0.5, 0])
    opt_meshes = create_tube(wp_opt, radius=0.003, color=[0, 0, 1])
    nom_meshes = create_tube(wp_nom, radius=0.002, color=[1, 0.5, 0])
    o3d.visualization.draw_geometries(
        [mesh, black_sphere] + nom_meshes + opt_meshes
    )
    # o3d.visualization.draw_geometries([
    #     mesh,
    #     nom_lines,
    #     opt_lines,
    # ])


if __name__ == "__main__":
    main()