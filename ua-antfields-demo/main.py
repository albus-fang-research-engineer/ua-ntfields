# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import argparse
from models import model_igibson as md

# Define modes
EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2


# Define the model path
scale_factor = 10
modelPath = './Experiments'

def main():
    parser = argparse.ArgumentParser(description='Train the model with or without exploration.')
    parser.add_argument('--no_explore', action='store_true', help='Disable exploration mode.')
    
    args = parser.parse_args()

    # Set the mode based on the --no_explore flag
    if args.no_explore:
        mode = READ_FROM_COOKED_DATA
    else:
        mode = EXPLORATION  # Default to exploration mode

    renderer = None
    if mode in [EXPLORATION]:
        from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
        meshpath = "data/mesh_superior_normalized.obj"
        
        renderer = MeshRenderer(width=1200, height=680)
        renderer.load_object(meshpath, scale=np.array([1, 1, 1]) * scale_factor)
        renderer.add_instance_group([0])
        camera_pose = np.array([0, 0, 1])    
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
        renderer.set_fov(90)

    # Initialize and train the model
    # model = md.Model(modelPath, 3, scale_factor, mode, renderer, device='cuda:0')
    # model.train()
    global_base = modelPath
    import os

    # ===== CREATE GLOBAL RUN FOLDER =====
    global_base = modelPath

    global_run_id = 0
    while True:
        global_folder = os.path.join(global_base, f"CHANCE_CONSTRAINED_GLOBAL_RUN_{global_run_id}")
        if not os.path.exists(global_folder):
            os.makedirs(global_folder)
            break
        global_run_id += 1

    print(f"Global run folder: {global_folder}")
    lengths = []
    num_runs = 10
    collisions = 0
    all_policy_times = []          # <-- add
    all_optimizer_times = []       # <-- add
    all_total_times = []           # <-- add
    for i in range(num_runs):
        print(f"\n===== Run {i+1}/{num_runs} =====")

        model = md.Model(global_folder, 3, scale_factor, mode, renderer, device='cuda:0')
        
        result = model.train()
        all_policy_times.extend(result["policy_times"])              # <-- add
        all_optimizer_times.extend(result["optimizer_times"])        # <-- add
        all_total_times.extend(result["total_planning_times"])       # <-- add
        if result["collision"]:
            collisions += 1
            print("Run ended with collision")
        elif result["length"] is not None:
            lengths.append(result["length"])
        else:
            print("Warning: run did not reach goal")
        # if path_length is not None:
        #     lengths.append(path_length)
        # else:
        #     print("Warning: run did not reach goal")

    lengths = np.array(lengths)
    policy_times = np.array(all_policy_times)
    optimizer_times = np.array(all_optimizer_times)
    total_times = np.array(all_total_times)
    print("\n===== FINAL RESULTS =====")
    print(f"Runs completed: {len(lengths)}")
    print(f"Mean path length: {np.mean(lengths) * 10:.4f} m")
    print(f"Std: {np.std(lengths)*10:.4f} m")
    print(f"Collision rate: {collisions / num_runs:.2f}")
    if len(total_times) > 0:
        print(f"Planning calls:        {len(total_times)}")
        print(f"Mean policy time:      {np.mean(policy_times)*1000:.2f} ms  (std {np.std(policy_times)*1000:.2f})")
        print(f"Mean optimizer time:   {np.mean(optimizer_times)*1000:.2f} ms  (std {np.std(optimizer_times)*1000:.2f})")
        print(f"Mean total plan time:  {np.mean(total_times)*1000:.2f} ms  (std {np.std(total_times)*1000:.2f})")
    else:
        print("No planning calls recorded.")

if __name__ == '__main__':
    main()
