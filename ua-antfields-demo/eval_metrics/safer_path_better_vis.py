import numpy as np
import open3d as o3d
import os

# ==== CONFIG ====
MESH_PATH = "/antfields/data/mesh.obj"
NPY_NOM_PATH = "/antfields/baseline_path_collision_step3.npy"
NPY_OPT_PATH = "/antfields/Experiments/03_21_12_19/traj_200.npy"
SAVE_PATH = "/antfields/rendered_figure.png"


def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Waypoint file not found: {path}")
    return np.load(path)


def ensure_3d(pts):
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    return pts


def create_tube(points, radius=0.03, color=[0, 0, 1]):
    meshes = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            continue

        direction = direction / length

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.paint_uniform_color(color)
        cylinder.compute_vertex_normals()

        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        if np.linalg.norm(v) > 1e-6:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
        else:
            # Handles parallel / anti-parallel case
            if c > 0:
                R = np.eye(3)
            else:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])

        cylinder.rotate(R, center=np.zeros(3))
        cylinder.translate((p1 + p2) / 2.0)
        meshes.append(cylinder)

    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.1)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere.translate(p)
        meshes.append(sphere)

    return meshes

def crop_mesh_by_z(mesh, z_max):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    tri_vertices = vertices[triangles]              # (T, 3, 3)
    tri_z_max = tri_vertices[:, :, 2].max(axis=1)   # max z per triangle

    keep = tri_z_max < z_max

    cropped = o3d.geometry.TriangleMesh()
    cropped.vertices = mesh.vertices
    cropped.triangles = o3d.utility.Vector3iVector(triangles[keep])
    cropped.remove_unreferenced_vertices()
    cropped.compute_vertex_normals()
    return cropped

def crop_mesh_to_path_region(mesh, path_points, margin=0.05):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    pmin = path_points.min(axis=0) - margin
    pmax = path_points.max(axis=0) + margin

    tri_verts = verts[tris]
    tri_centers = tri_verts.mean(axis=1)

    keep = np.all((tri_centers >= pmin) & (tri_centers <= pmax), axis=1)

    cropped = o3d.geometry.TriangleMesh()
    cropped.vertices = mesh.vertices
    cropped.triangles = o3d.utility.Vector3iVector(tris[keep])
    cropped.remove_unreferenced_vertices()
    cropped.compute_vertex_normals()
    return cropped

def setup_camera(vis, lookat, front, up, zoom=0.7):
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)


def main():
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh.compute_vertex_normals()
    mesh = crop_mesh_by_z(mesh, z_max=0.09)
    
    mesh.paint_uniform_color([0.82, 0.82, 0.82])  # light gray mesh

    wp_opt = ensure_3d(load_waypoints(NPY_OPT_PATH))
    wp_nom = ensure_3d(load_waypoints(NPY_NOM_PATH))

    print(f"[INFO] Optimized shape: {wp_opt.shape}")
    print(f"[INFO] Nominal shape: {wp_nom.shape}")
    mesh = crop_mesh_to_path_region(mesh, wp_opt, margin=0.1)
    robot_pose = wp_nom[-4]
    start_pt = wp_nom[0]
    goal_pt = wp_nom[-1]
    start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.013)
    start_marker.paint_uniform_color([0.0, 0.8, 0.0])   # green
    start_marker.compute_vertex_normals()
    start_marker.translate(start_pt)

    goal_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.013)
    goal_marker.paint_uniform_color([0.8, 0.0, 0.8])   # magenta
    goal_marker.compute_vertex_normals()
    goal_marker.translate(goal_pt)
    robot_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    robot_marker.paint_uniform_color([1.0, 0, 0])
    robot_marker.compute_vertex_normals()
    robot_marker.translate(robot_pose)

    # Make paths a bit thicker for figure quality
    opt_meshes = create_tube(wp_opt, radius=0.0045, color=[0.1, 0.35, 0.95])
    nom_meshes = create_tube(wp_nom, radius=0.0035, color=[1.0, 0.55, 0.1])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1800, height=1200, visible=True)

    vis.add_geometry(mesh)
    vis.add_geometry(robot_marker)
    vis.add_geometry(start_marker)
    vis.add_geometry(goal_marker)
    # vis.add_geometry(frame)  

    for g in nom_meshes:
        vis.add_geometry(g)
    for g in opt_meshes:
        vis.add_geometry(g)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # white background
    render_opt.light_on = True
    render_opt.mesh_show_back_face = True

    # Camera target = center between both paths
    all_pts = np.vstack([wp_nom, wp_opt])
    center = all_pts.mean(axis=0)

    # Try a nice angled view
    setup_camera(
        vis,
        lookat=center,
        front=[-0.45, -0.35, 0.95],
        up=[0.0, 0.0, 1.0],
        zoom=0.65,
    )

    vis.poll_events()
    vis.update_renderer()

    # Save directly from renderer
    vis.capture_screen_image(SAVE_PATH, do_render=True)
    print(f"[INFO] Saved render to: {SAVE_PATH}")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()