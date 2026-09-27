# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import torch
from dataprocessing import isdf_sample
from test_igib import viz
import matplotlib.pyplot as plt
from load_njsdf.inference import (
    load_sdf_2d_model,
    risk_distance_cvar
)

def voxel_downsample(pc, voxel_size):
    coords = torch.floor(pc / voxel_size)
    unique, idx = torch.unique(coords, dim=0, return_inverse=True, return_counts=False, sorted=False)
    return pc[idx]

def knn_local(query_pts, surf_pc, K):
    dists = torch.cdist(query_pts, surf_pc)
    knn_dists, knn_idx = torch.topk(dists, K, largest=False)
    return surf_pc[knn_idx]

def radial_downsample(points, center, max_pts=3000, near_radius=1.0):
    """
    points : (N,3) torch
    center : (3,)  torch

    Keeps:
    - all near points
    - random far points

    returns: (M,3)
    """
    d = torch.norm(points - center, dim=1)

    near_mask = d < near_radius
    near_pts = points[near_mask]

    far_pts = points[~near_mask]

    remaining = max(0, max_pts - near_pts.shape[0])

    if far_pts.shape[0] > remaining:
        idx = torch.randperm(far_pts.shape[0], device=points.device)[:remaining]
        far_pts = far_pts[idx]

    return torch.cat([near_pts, far_pts], dim=0)

def batched_min_mean_distance(model, query_pts, local_obs):

    device = model.dist_device

    M, K, _ = local_obs.shape

    # Expand robot positions to match obstacle batch
    robot_xy = query_pts[:, None, :2].expand(-1, K, -1)
    obs_xy   = local_obs[:, :, :2]

    # Network input: (M*K, 4)
    net_input = torch.cat([robot_xy, obs_xy], dim=-1)
    net_input = net_input.reshape(M*K, 4)

    from load_njsdf.inference import predict_mu_var

    # We only care about mu
    mu, _ = predict_mu_var(model.dist_model, net_input)

    # Reshape back to (M, K)
    mu = mu.view(M, K)

    # Take minimum mean distance per query point
    min_mu = mu.min(dim=1).values

    return min_mu

def batched_cvar_distance(model, query_pts, local_obs, alpha=0.45, tail=0.1):

    device = model.dist_device

    M, K, _ = local_obs.shape

    robot_xy = query_pts[:, None, :2].expand(-1, K, -1)
    obs_xy   = local_obs[:, :, :2]

    net_input = torch.cat([robot_xy, obs_xy], dim=-1)
    net_input = net_input.reshape(M*K, 4)

    from load_njsdf.inference import predict_mu_var

    mu, var = predict_mu_var(model.dist_model, net_input)

    sigma = torch.sqrt(torch.clamp(var, min=1e-12))

    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device)
    )

    z = normal.icdf(torch.tensor(alpha, device=device))

    var_i = mu + z * sigma
    var_i = var_i.view(M, K)

    k = max(1, int(np.ceil(tail * K)))
    worst = torch.topk(var_i, k, largest=False).values

    return worst.mean(dim=1)


def interpolate_np_arrays(start_array, end_array, num_steps):
    lin_int = np.linspace(0, 1, num=num_steps)[:, None]
    return start_array[None, :] * (1 - lin_int) + end_array[None, :] * lin_int

def get_current_rgb_frame(renderer, hidden_instances) -> np.ndarray:
    frames = renderer.render(modes=("rgb",), hidden=hidden_instances)
    assert len(frames) == 1
    rgb_frame = frames[0]
    assert len(rgb_frame.shape) == 3 and rgb_frame.shape[-1] == 4, f"Unexpected RGB frame shape: {rgb_frame.shape}"
    rgb_frame = (rgb_frame[:, :, :3] * 255).astype(np.uint8)
    return rgb_frame

def get_current_lidar_frame(renderer, camera_position, hidden_instance=[]) -> np.ndarray:
    from igibson.robots.robot_base import BaseRobot
    from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
    # def get_lidar_from_depth_my(renderer):
    #     lidar_readings = renderer.render(modes=("3d"))[0]
    
    # hidden_instances = env.scene.robots[0].renderer_instances

    ##############lidar with hidden objects##############
    def get_lidar_from_depth(renderer):
        """
        Get partial LiDAR readings from depth sensors with limited FOV.

        :return: partial LiDAR readings with limited FOV
        """
        # renderer = env.simulator.renderer
        lidar_readings = renderer.render(modes=("3d"), hidden=hidden_instance)[0]
        # print("lidar reading shape 0",lidar_readings[0].shape)
        # filter = lidar_readings[0][:, :, :] != [0, 0, 0, 1]
        # print("filter shape",lidar_readings[0][filter])
        # print("lidar reading shape 1",lidar_readings[1].shape)

        # print("renderer.x_samples",renderer.x_samples)
        # print("renderer.y_samples",renderer.y_samples)
        lidar_readings = lidar_readings[renderer.x_samples, renderer.y_samples, :3]
        # lidar_readings = lidar_readings[:, :, :3]
        dist = np.linalg.norm(lidar_readings, axis=1)
        # print("lidar reading shape 2",lidar_readings.shape)
        lidar_readings = lidar_readings[dist > 0]

        dist = dist[dist > 0]
        dist_m = dist
        # compute sigma in meters
        sigma_m = np.where(dist_m < 1.0, 0.01, 0.01 * dist_m)
        # convert sigma back to mesh units
        sigma = sigma_m
        dirs = lidar_readings / dist[:, None]
        range_noise = np.random.randn(len(dist)) * sigma
        lidar_readings = lidar_readings + dirs * range_noise[:, None]


        lidar_readings[:, 2] = -lidar_readings[:, 2]  # make z pointing out
        # print("lidar reading shape 3",lidar_readings.shape)
        return lidar_readings

    ############lidar code start############
    def get_lidar_all_new(renderer, offset_with_camera=np.array([0, 0, 0])):
        """
        Get complete LiDAR readings by patching together partial ones.

        :param offset_with_camera: optionally place the lidar scanner
            with an offset to the camera
        :return: complete 360 degree LiDAR readings
        """
        # for instance in renderer.instances:
        #     if isinstance(instance.ig_object, BaseRobot):
        #         camera_pos = instance.ig_object.eyes.get_position() + offset_with_camera
        #         print("camera_pos", camera_pos)
        #         orn = instance.ig_object.eyes.get_orientation()
        #         mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
        #         view_direction = mat.dot(np.array([1, 0, 0]))
        #         up_direction = mat.dot(np.array([0, 0, 1]))
        #         renderer.set_camera(camera_pos, camera_pos + view_direction, up_direction)
        # else: #meaning no robot in the scene only mesh
        camera_pos = offset_with_camera
        view_dirction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        renderer.set_camera(camera_pos, camera_pos + view_dirction, up_direction)

        original_fov = renderer.vertical_fov
        renderer.set_fov(120) #TODO: change fov later 113
        lidar_readings = []
        r = np.array(
            [
                [
                    np.cos(-np.pi / 5),
                    0,
                    -np.sin(-np.pi / 5),
                    0,
                ],
                [0, 1, 0, 0],
                [np.sin(-np.pi / 5), 0, np.cos(-np.pi / 5), 0],
                [0, 0, 0, 1],
            ]
        )

        transformation_matrix = np.eye(4)
        for i in range(10):
            lidar_one_view = get_lidar_from_depth(renderer)
            lidar_readings.append(lidar_one_view.dot(transformation_matrix[:3, :3]))
            renderer.V = r.dot(renderer.V)
            transformation_matrix = np.linalg.inv(r).dot(transformation_matrix)

        lidar_readings = np.concatenate(lidar_readings, axis=0)
        # currently, the lidar scan is in camera frame (z forward, x right, y up)
        # it seems more intuitive to change it to (z up, x right, y forward)
        lidar_readings = lidar_readings.dot(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

        # -------------------------------------------------
        # Open3D filtering
        # -------------------------------------------------

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_readings)

        # optional: downsample first (helps remove duplicates)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # statistical outlier removal
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=36,
            std_ratio=0.8
        )

        lidar_readings = np.asarray(pcd.points)

        # -------------------------------------------------
        renderer.set_fov(original_fov)
        return lidar_readings   
    ############lidar code end############
    return get_lidar_all_new(renderer, camera_position)

def get_c2w_transform(renderer) -> np.ndarray:
    def cvt_transform_to_opengl_format(transform) -> np.ndarray:
        """
        Convert camera-space-2-world-space transform into OpenGL format (for iSDF)
        """
        if transform is None:
            transform = np.eye(4)
        T = np.array([[1, 0, 0, 0],
                    [0, np.cos(np.pi), -np.sin(np.pi), 0],
                    [0, np.sin(np.pi), np.cos(np.pi), 0],
                    [0, 0, 0, 1]])
        return transform @ T
    c2w_transform = np.linalg.inv(renderer.V)
    c2w_transform = cvt_transform_to_opengl_format(c2w_transform)  # expected by iSDF
    return c2w_transform

def get_current_rgbl_framedata_for_camera_params(renderer, camera_pose, target, up, hidden_instances):
    renderer.set_camera(camera_pose, target, up, cache=False)
    rgb = get_current_rgb_frame(renderer, hidden_instances)
    depth_img = get_current_lidar_frame(renderer,  camera_position=camera_pose)
    c2w_transform_mat = get_c2w_transform(renderer)
    return rgb, depth_img, c2w_transform_mat

def get_lidar_and_transform_from_pose(renderer, camera_pose):
    depth_img = get_current_lidar_frame(renderer,  camera_position=camera_pose)
    c2w_transform_mat = get_c2w_transform(renderer)
    return depth_img, c2w_transform_mat

def sample_points_and_speeds_from_pos(renderer, position, minimum, maximum, num=10000, scale_factor=1):
    sample_pts = sample_points_from_pos(renderer, position, scale_factor)
    points, bounds = get_bounds_from_pts(sample_pts)

    pc = points.view(-1, 3)
    bounds = bounds.view(-1, 1)

    minimum *= scale_factor
    maximum *= scale_factor
    points, speeds, bounds = sample_points_and_speeds_from_bounds(pc, bounds, minimum=minimum, maximum=maximum, num=num)

    #! we need to scale the points and bounds back to the original scale
    points /= scale_factor
    bounds /= scale_factor
    
    return points, speeds, bounds

def sample_points_and_speeds_from_pos_new(model, position, minimum, maximum, num=10000, scale_factor=1):
    sample_pts = sample_points_from_pos(model, position, scale_factor)
    #points, bounds = get_bounds_from_pts(sample_pts)
    #'''
    points, bounds = get_bounds_from_pts(sample_pts)

    points = points.view(-1, 3)
    bounds = bounds.view(-1, 1)
    #'''

    #speeds = torch.clip(bounds, minimum, maximum)/maximum

    minimum *= scale_factor
    maximum *= scale_factor

    valid_indices = torch.where((bounds < maximum+0.016) & (bounds > minimum-0.005))[0] 

    x0 = points[valid_indices]
    y0 = torch.clip(bounds[valid_indices], minimum, maximum)/maximum

    dP = torch.rand((x0.shape[0],3),dtype=torch.float32, device='cuda')-0.5
    rL = (torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda'))*1.5
    x1 = x0 + torch.nn.functional.normalize(dP,dim=1)*rL

    position = torch.tensor(position).cuda()

    ray0 = x1 - position*scale_factor
    ray1 = sample_pts["surf_pc"] - position*scale_factor
    norm0 = ray0.norm(dim=-1)
    norm1 = ray1.norm(dim=-1)
    dot = (ray0/norm0.unsqueeze(1))@(ray1/norm1.unsqueeze(1)).T
    dot, closest_ixs1 = dot.max(axis=-1)
    # print(dot)

    valid_indices1 = torch.where( (norm0 < norm1[closest_ixs1])&(dot>0.995)&(norm0<2.5))[0]
    #print(dot)
    x0 = x0[valid_indices1]
    x1 = x1[valid_indices1]
    y0 = y0[valid_indices1]

    diff1 = x1.unsqueeze(1) - sample_pts["surf_pc"] 
    dists1 = diff1.norm(dim=-1)
    dists1, closest_ixs1 = dists1.min(axis=-1)
    # dists1 -= 0.02
    y1 = torch.clip(dists1, minimum, maximum)/maximum

    # print(closest_ixs1)

    points = torch.cat((x0, x1), dim=1)
    speeds = torch.cat((y0, y1.unsqueeze(1)), dim=1)
    bounds = torch.cat((bounds[valid_indices][valid_indices1], dists1.unsqueeze(1)), dim=1)

    points /= scale_factor
    bounds /= scale_factor
    obstacle_points = sample_pts["surf_pc"] / scale_factor
    #if False and is_gt_speed: #ground truth
    #    bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
    # print(bounds)
    #points, speeds, bounds = sample_points_and_speeds_from_bounds(pc, bounds, minimum=minimum, maximum=maximum, num=num)
    
    return points[0:3000], speeds[0:3000], bounds[0:3000]

def sample_points_and_speeds_from_pos_neural(model, position, minimum, maximum, num=10000, scale_factor=1):
    sample_pts = sample_points_from_pos(model, position, scale_factor)
    #points, bounds = get_bounds_from_pts(sample_pts)
    #'''
    points, bounds = get_bounds_from_pts(sample_pts)

    points = points.view(-1, 3)
    bounds = bounds.view(-1, 1)
    #'''

    #speeds = torch.clip(bounds, minimum, maximum)/maximum

    minimum *= scale_factor
    maximum *= scale_factor

    valid_indices = torch.where((bounds < maximum+0.0) & (bounds > minimum-0.000))[0] 

    x0 = points[valid_indices]
    # y0 = torch.clip(bounds[valid_indices], minimum, maximum)/maximum

    dP = torch.rand((x0.shape[0],3),dtype=torch.float32, device='cuda')-0.5
    rL = (torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda'))*1.5
    x1 = x0 + torch.nn.functional.normalize(dP,dim=1)*rL

    position = torch.tensor(position).cuda()

    ray0 = x1 - position*scale_factor
    ray1 = sample_pts["surf_pc"] - position*scale_factor
    norm0 = ray0.norm(dim=-1)
    norm1 = ray1.norm(dim=-1)
    dot = (ray0/norm0.unsqueeze(1))@(ray1/norm1.unsqueeze(1)).T
    dot, closest_ixs1 = dot.max(axis=-1)
    # print(dot)

    valid_indices1 = torch.where( (norm0 < norm1[closest_ixs1])&(dot>0.995)&(norm0<2.5))[0]
    #print(dot)
    x0 = x0[valid_indices1]
    x1 = x1[valid_indices1]
    # y0 = y0[valid_indices1]

#----------------------------------------------------------
#---------------- Calculate X1 distance -------------------
#----------------------------------------------------------
    # diff1 = x1.unsqueeze(1) - sample_pts["surf_pc"] 
    # dists1 = diff1.norm(dim=-1)
    # dists1, closest_ixs1 = dists1.min(axis=-1)

    surf_pc = sample_pts["surf_pc"] # 10x environment scale
    # --- downsample once ---
    # surf_pc = voxel_downsample(surf_pc, voxel_size=0.02)
    # surf_pc = radial_downsample(
    #     surf_pc,
    #     position * scale_factor,
    #     max_pts=3000,
    #     near_radius= maximum  * 1.5
    # )
    with torch.no_grad():
        # -------- neural distance for x0 (same as x1) --------
        local_obs_x0 = knn_local(x0, surf_pc, K=32)
        # dists0 = batched_cvar_distance(model, x0, local_obs_x0)
        # dists0 = batched_min_mean_distance(model, x0 / scale_factor, local_obs_x0 / scale_factor) * scale_factor
        dists0 = batched_cvar_distance(model, x0 / scale_factor, local_obs_x0 / scale_factor) * scale_factor
        y0 = torch.clip(dists0, minimum, maximum) / maximum
        # --- KNN for all query points at once ---
        local_obs = knn_local(x1, surf_pc, K=32)
        # --- neural CVaR distance ---
        # dists1 = batched_cvar_distance(model, x1, local_obs)
        # dists1 = batched_min_mean_distance(model, x1 / scale_factor, local_obs / scale_factor) * scale_factor
        dists1 = batched_cvar_distance(model, x1 / scale_factor, local_obs / scale_factor) * scale_factor
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
    # dists1 -= 0.02
    y1 = torch.clip(dists1, minimum, maximum)/maximum

    # print(closest_ixs1)

    points = torch.cat((x0, x1), dim=1)
    speeds = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1)), dim=1)
    bounds = torch.cat((bounds[valid_indices][valid_indices1], dists1.unsqueeze(1)), dim=1)

    points /= scale_factor
    bounds /= scale_factor
    obstacle_points = surf_pc / scale_factor
    #if False and is_gt_speed: #ground truth
    #    bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
    # print(bounds)
    #points, speeds, bounds = sample_points_and_speeds_from_bounds(pc, bounds, minimum=minimum, maximum=maximum, num=num)
    
    # return points[0:5000], speeds[0:5000], bounds[0:5000], obstacle_points
    return points[0:2066], speeds[0:2066], bounds[0:2066], obstacle_points

def sample_points_from_pos(model, position, scale_factor=1):
    """
    configs
    """
    n_rays = 6000 #5000
    dist_behind_surf = 0. #0.2
    n_strat_samples = 20 # 20
    n_surf_samples = 8  #8
    min_depth = 0.01
    max_depth = 0.25
    dist_behind_surf *= scale_factor
    min_depth *= scale_factor
    max_depth *= scale_factor

    device = 'cuda:0'
    ##! The following lidar sample code is at a large scale
    lidar, T_np = get_lidar_and_transform_from_pose(model.renderer, position*scale_factor)
    depth_dirs, depth = lidar, np.linalg.norm(lidar, axis=-1)
    rotation_matrix_90_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    depth_dirs = depth_dirs.dot(rotation_matrix_90_z.T)/depth[:, None]

    surf_pc = T_np[:3, 3] + depth_dirs*depth[:, None]  
    surf_pc /= scale_factor  

    if model.mode == 1:
        # import igl 
        # sqrD, ind, C = igl.point_mesh_squared_distance(surf_pc, model.v, model.f)
        # normals = model.normals[ind]
        # print("normals", normals.shape)
    
    # assert 1==0

        #TODO: save depth_dirs, depth, T_np, and current position to a dictionary
        frame_idx = model.frame_idx
        # savepath = model.folder+"/frame_"+str(frame_idx)+".npy"
        sub_dataset = {}
        sub_dataset["depth_dirs"] = depth_dirs
        sub_dataset["depth"] = depth
        sub_dataset["T_np"] = T_np
        # sub_dataset["normals"] = normals/np.linalg.norm(normals, axis=-1, keepdims=True)
        # sub_dataset["position"] = position*scale_factor
        # np.save(savepath, sub_dataset)


    depth = torch.from_numpy(depth[None,:]).float().to(device)
    T = torch.from_numpy(T_np[None,:]).float().to(device)
    # depth_dirs_np = sub_dataset["depth_dirs"][None, ...]
    depth_dirs = torch.from_numpy(depth_dirs[None,:]).float().to(device)
    sample_pts = isdf_sample.sample_lidar_points(depth, T, n_rays, depth_dirs, dist_behind_surf, n_strat_samples, n_surf_samples, min_depth, max_depth, device=device)

    return sample_pts

def get_bounds_from_pts(sample_pts):
    #? this usually is in large scale
    #? use new bound method
    diff = sample_pts["pc"][:, :, None] - sample_pts["surf_pc"] 
    # print("surf_pc", sample_pts["surf_pc"])
    dists = diff.norm(dim=-1)
    dists, closest_ixs = dists.min(axis=-1)
    behind_surf = sample_pts["z_vals"] > sample_pts["depth_sample"][:, None]
    dists[behind_surf] *= -1

    #!: sample pts include pc, z_vals, surf_pc, depth_sample. If bounds is smaller than some threshold, then the ray samples following this point will not be used.
    valid_threshold = 0.002
    filtered_pc_list = []
    filtered_bounds_list = []
    for i in range(dists.shape[0]):
        small_index_ls = (dists[i] <= valid_threshold).nonzero(as_tuple=True)[0]
        if len(small_index_ls) == 0:
            # meaning all the bounds are larger than the threshold
            filtered_pc_list.append(sample_pts["pc"][i])
            filtered_bounds_list.append(dists[i])
        else:
            # meaning some bounds are smaller than the threshold, then we only use the points before the first small bound
            filtered_pc_list.append(sample_pts["pc"][i, :small_index_ls[0]])
            filtered_bounds_list.append(dists[i, :small_index_ls[0]])
    filtered_pc = torch.cat([ray_points for ray_points in filtered_pc_list if ray_points.nelement() > 0], dim=0)
    filtered_bounds = torch.cat([ray_bounds for ray_bounds in filtered_bounds_list if ray_bounds.nelement() > 0], dim=0)

    # dists -= 0
    # filtered_bounds -= 0.02 #! subtract some value to make the bounds smaller
    filtered_bounds -= 0.00001
    #? add a ceiling and floor to the bounds
    # dist_ceil = torch.abs(sample_pts["pc"][:, :, 2]-2.3)
    # dist_floor = torch.abs(sample_pts["pc"][:, :, 2]+0.02)
    # dists = torch.min(torch.min(dists, dist_ceil), dist_floor)
    bounds = dists
    # return sample_pts["pc"], bounds
    return filtered_pc, filtered_bounds


def sample_points_and_speeds_from_bounds(pc, bounds, minimum=0.1, maximum=2.0, num=10000):
    # speeds = torch.clip((bounds - minimum) / (maximum - minimum), 0.1, 1)
    speeds = torch.clip(bounds, minimum, maximum)/maximum

    valid_indices = torch.where((speeds < 1) & (speeds > minimum/maximum))[0] 

    if num <= len(valid_indices):
        # Select without replacement if num is less than or equal to the size of valid_indices
        start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
        # end_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
    else:
        # Select with replacement if num is greater than the size of valid_indices
        rand_indices = torch.randint(0, len(valid_indices), (num,))
        start_indices = valid_indices[rand_indices]

        rand_indices = torch.randint(0, len(valid_indices), (num,))
        # end_indices = valid_indices[rand_indices]

    end_indices = torch.randint(0, pc.shape[0], (num,))
    x0 = pc[start_indices]
    x1 = pc[end_indices]

    x = torch.cat((x0, x1), dim=1)
    y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)
    z = torch.cat((bounds[start_indices], bounds[end_indices]), dim=1)
    return x, y, z

def main():
    import yaml 
    from igibson.envs.igibson_env import iGibsonEnv
    """
    1. load environment in igibson
    2. some dataset to convert collected data
    3. train model
    """
    config_file = "configs/config-simple-room-hires.yaml"
    config_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    config_data["hide_robot"] = True
    config_data["texture_randomization_freq"] = None
    config_data["object_randomization_freq"] = None

    # choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"]
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive",
                     action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0,
                     use_pb_gui=False)

    # env.simulator.viewer.initial_pos = [1., 0.7, 1.7]
    # env.simulator.viewer.initial_view_direction = [-0.5, -0.9, -0.5]
    # env.simulator.viewer.reset_viewer()


    # # fix position of all objects current in env
    # for id in env.simulator.scene.get_body_ids():
    #     pb.changeDynamics(bodyUniqueId=id, linkIndex=-1, mass=0)
    env.simulator.step()
    robot = env.scene.robots[0]
    hidden_instances = robot.renderer_instances

    #! only for explore scene
    if False:
        while True:
            env.simulator.step()

    # ! export scene mesh
    if False:
        vertices, faces = env.simulator.renderer.dump()
        path = "Beechwood_0_int_scene_mesh_1.obj"
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(path)

    # ! camera trajectory
    height = 0.2
    camera_pose_dir_pairs = [
        [(-5, 4.7, height), (0, -1, 0)],
        [(-5.5, 3.2, height), (-0.7, -0.7, 0)],
        [(-6.8, 3.6, height), (-0.8, 0.6, 0)],
        [(-8.8, 1.6, height), (-0.6, -0.8, 0)],
        [(-9.9, 0.3, height), (-0.7, -0.8, 0)],
        [(-10.3, -2.3, height), (0.1, -1, 0)],
        [(-8.1, -2.3, height), (1, -0.0, 0)],
        [(-7.9, 0, height), (0, 1, 0)],
        [(-9.2, 0.9, height), (-0.7, 0.7, 0)],
        [(-7.5, 3.1, height), (0.6, 0.8, 0)],
        [(-4, 1.6, height), (0.7, -0.7, 0)],
        [(-1.9, 1, height), (0.9, -0.3, 0)],
        [(-1.1, 0.2, height), (0.4, -0.9, 0)],
        [(0, 0, height), (1, 0, 0)],
        [(-1.4, 1, height), (-0.8, 0.5, 0)],
        [(-2.4, 2.2, height), (-0.5, 0.9, 0)],
        [(-2.4, 2.2, height), (1, -0.3, 0)],
        [(-1.6, 2, height), (1.0, -0.1, 0)],
        [(-1.3, 2.3, height), (0.3, 0.9, 0)],
        [(-3.9, -0.7, height), (-0.9, -0.5, 0)],
        [(-3.4, -1.3, height), (0, -1, 0)],
        [(-2.7, -2.7, height), (0.9, -0.4, 0)],
        [(-0.1, -3.3, height), (0.5, -0.8, 0)],
        [(-2, -2.7, height), (-0.9, 0.5, 0)],
        [(-6.8, -3, height), (-1, 0, 0)],
        [(-6.8, -5, height), (-0.1, -1, 0)],
        [(-4.8, -5, height), (1, 0, 0)],
        [(-3.6, -3.5, height), (0.6, 0.8, 0)],
    ]

    # up = [0, 0, 1]
    # frame_count = 1
    # for i in range(1, len(camera_pose_dir_pairs)):
    #     camera_pose_prev, dir_prev = camera_pose_dir_pairs[i - 1]
    #     camera_pose_cur, dir_cur = camera_pose_dir_pairs[i]

    #     pose_prev = np.array(camera_pose_prev)
    #     dir_prev = np.array(dir_prev)
    #     target_prev = pose_prev + dir_prev

    #     pose_cur = np.array(camera_pose_cur)
    #     dir_cur = np.array(dir_cur)
    #     target_cur = pose_cur + dir_cur

    #     # Use numpy for interpolation
    #     camera_poses = interpolate_np_arrays(pose_prev, pose_cur, frame_count)
    #     targets = interpolate_np_arrays(target_prev, target_cur, frame_count)

    #     # Extract frames
    #     for frame_idx in range(frame_count):
    #         camera_pose = camera_poses[frame_idx]
    #         target = targets[frame_idx]

    #         # frame_data = get_current_rgbd_framedata_for_camera_params(env, camera_pose, target, up, hidden_instances)
    #         frame_data = get_lidar_and_transform_from_pose(env, camera_pose)

    #! visualize
    # points, speeds = sample_points_and_speeds_from_pos(env, np.array([0, 0, height]))
    # points = points.cpu().numpy()
    # speeds = speeds.cpu().numpy()
    # np.save("mypoints.npy", points)
    # np.save("myspeeds.npy", speeds)



    #! train model
    from models import model_igibson_online_continual as md
    modelPath = './Experiments/active'

    model    = md.Model(modelPath, env, 3, device='cuda:0')

    model.train()

def transform_to_small_scene(points):
    points[:, 0] += 4.2529825
    points[:, 1] -= 2.682928
    points[:, 2] -= 1.1388535

    points /= 9.757375
    return points

def main2():
    from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
    # meshpath = "mesh_z_up_11_28.obj"
    meshpath = "mesh_scaled_11_29_15.obj" #xiao
    renderer = MeshRenderer(width=1200, height=680)
    scale_factor = 10
    renderer.load_object(meshpath, scale=np.array([1, 1, 1])*scale_factor)
    renderer.add_instance_group([0])
    camera_pose = np.array([0, 1, 0])    
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
    renderer.set_fov(90)

    if True: #my way of sample points and speeds
        # camera_positions = camera_positions[89:90, :]
        camera_positions = np.array([[0, 0.02, 0]])
        for i in range(camera_positions.shape[0]):
            points, speeds, bounds = sample_points_and_speeds_from_pos_new(renderer, camera_positions[i], minimum=0.005, maximum=0.05, num=5000, scale_factor=scale_factor)

            if i == 0:
                all_points = points
                all_speeds = speeds
            else:
                all_points = torch.cat((all_points, points), dim=0)
                all_speeds = torch.cat((all_speeds, speeds), dim=0)

        local_index = torch.randperm(all_points.shape[0])[:int(all_points.shape[0]*0.3)]
        
        all_start = all_points[:, :3]
        all_end = all_points[:, 3:]
        start_speeds = all_speeds[:, 0]
        end_speeds = all_speeds[:, 1]
        start_indices = torch.randperm(all_points.shape[0])[:int(all_points.shape[0]*0.7)]
        end_indices = torch.randperm(all_points.shape[0])[:int(all_points.shape[0]*0.7)]
        all_start = all_start[start_indices]
        all_end = all_end[end_indices]
        start_speeds = start_speeds[start_indices]
        end_speeds = end_speeds[end_indices]

        local_points_combination = all_points[local_index]
        local_speeds_combination = all_speeds[local_index]
        global_points_combination = torch.cat((all_start, all_end), dim=1)
        global_speeds_combination = torch.cat((start_speeds[:,None], end_speeds[:,None]), dim=1)

        all_points = torch.cat((local_points_combination, global_points_combination), dim=0)
        all_speeds = torch.cat((local_speeds_combination, global_speeds_combination), dim=0)

        print("all_points:", all_points.shape)
        print("all_speeds:", all_speeds.shape)
        np.save("mypoints.npy", all_points.cpu().numpy())
        np.save("myspeeds.npy", all_speeds.cpu().numpy())
    else: #sample lidar points
        n_rays = 5000
        dist_behind_surf = 0.01
        n_strat_samples = 20
        n_surf_samples = 8
        min_depth = 0.01
        max_depth = 5

        all_points = []
        all_bounds = []
        device = 'cuda:0'
        position = [0, 0, 0.4]
        all_surf_pc = []
        for position in camera_positions:
            lidar, T_np = get_lidar_and_transform_from_pose(renderer, position)
            depth_dirs, depth = lidar, np.linalg.norm(lidar, axis=-1)
            rotation_matrix_90_z = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            depth_dirs = depth_dirs.dot(rotation_matrix_90_z.T)/depth[:, None]

            depth = torch.from_numpy(depth[None,:]).float().to(device)
            T = torch.from_numpy(T_np[None,:]).float().to(device)
            # depth_dirs_np = sub_dataset["depth_dirs"][None, ...]
            depth_dirs = torch.from_numpy(depth_dirs[None,:]).float().to(device)
            surf_pc = isdf_sample.surface_sample(T, depth_dirs, depth, device)

            surf_pc = transform_to_small_scene(surf_pc)
            all_surf_pc.append(surf_pc.cpu().numpy())
        all_surf_pc = np.vstack(all_surf_pc)
        np.save("surf_pc.npy", all_surf_pc)

if __name__ == "__main__":
    # main()
    main2()
