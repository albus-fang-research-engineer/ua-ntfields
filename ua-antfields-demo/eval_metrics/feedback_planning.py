import numpy as np
import open3d as o3d
import os

# ==== CONFIG ====
MESH_PATH = "/antfields/data/mesh.obj"

NPY_OPT_PATH = "/antfields/Experiments/03_19_15_42/epoch_0100_optimized.npy"
NPY_NOM_PATH = "/antfields/Experiments/03_19_15_42/epoch_0050_optimized.npy"
NPY_THIRD_PATH = "//antfields/Experiments/03_19_15_42/epoch_0150_optimized.npy"

def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Waypoint file not found: {path}")
    return np.load(path)


def ensure_3d(pts):
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    return pts

import open3d.visualization.rendering as rendering

def add_with_alpha(vis, name, geom, color, alpha):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [color[0], color[1], color[2], alpha]
    vis.add_geometry(name, geom, mat)
    
def create_lineset(points, color):
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_start_marker(point, radius=0.0105, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(point)
    return sphere

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

def remove_ceiling(mesh, z_threshold=0.5):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # Keep triangles ONLY if all vertices are below threshold
    mask = np.all(verts[tris][:, :, 2] < z_threshold, axis=1)

    mesh.triangles = o3d.utility.Vector3iVector(tris[mask])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh
def main():
    import open3d.visualization.gui as gui
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh = remove_ceiling(mesh, z_threshold=0.10)
    mesh.compute_vertex_normals()

    wp_opt = ensure_3d(load_waypoints(NPY_OPT_PATH))
    wp_nom = ensure_3d(load_waypoints(NPY_NOM_PATH))
    wp_third = ensure_3d(load_waypoints(NPY_THIRD_PATH))
    y_offset = -0.005
    wp_opt[-3:-1, 1] += y_offset
    print(f"[INFO] Optimized shape: {wp_opt.shape}")
    print(f"[INFO] Nominal shape: {wp_nom.shape}")
    print(f"[INFO] Third shape: {wp_third.shape}")
    # 🔵 Optimized
    # opt_lines = create_lineset(wp_opt, [0, 0, 1])
    start_opt = create_start_marker(wp_opt[0], radius=0.0105, color=[0, 0, 0])
    start_nom = create_start_marker(wp_nom[0], radius=0.0105, color=[0, 0, 0])
    start_third = create_start_marker(wp_third[0], radius=0.0105, color=[0, 0, 0])
    # # 🟠 Nominal
    # nom_lines = create_lineset(wp_nom, [1, 0.5, 0])
    opt_meshes = create_tube(wp_opt, radius=0.003, color=[0, 0, 1])
    nom_meshes = create_tube(wp_nom, radius=0.0026, color=[1, 0, 0])
    # 🟢 Third path (green)
    third_meshes = create_tube(wp_third, radius=0.0036, color=[0, 1, 0])
    # o3d.visualization.draw_geometries(
    #     [mesh] + nom_meshes + opt_meshes + third_meshes + [start_nom, start_third, start_opt]
    # )
    # ---- initialize GUI ----
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Paths", 1024, 768)
    vis.set_background(
        np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),  # black
        None
    )
    # mesh (solid)
    vis.add_geometry("mesh", mesh)
    vis.show_skybox(False)
    vis.scene.scene.enable_sun_light(False)
    # 🔴 nominal (faded)
    for i, m in enumerate(nom_meshes):
        add_with_alpha(vis, f"nom_{i}", m, [1, 0, 0], 0.5)

    # 🔵 optimized (semi solid)
    for i, m in enumerate(opt_meshes):
        add_with_alpha(vis, f"opt_{i}", m, [0, 0, 1], 0.7)

    # 🟢 third (FULL emphasis)
    for i, m in enumerate(third_meshes):
        add_with_alpha(vis, f"third_{i}", m, [0, 1, 0], 1.0)
    for i, s in enumerate([start_nom, start_third, start_opt]):
        add_with_alpha(vis, f"start_{i}", s, [0, 0, 0], 1.0)
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    main()