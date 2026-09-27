# Uncertainty-Aware NTField

This repository contains the uncertainty-aware Neural Time Field (NTField) training pipeline.

## Docker Setup

From the repository root, enter the devcontainer folder:

```bash
cd .devcontainer
```

Before starting the container, open `.devcontainer/docker-compose.yml` and, on **line 19**, change the directory before the `:` to the actual path of this folder on your machine. This folder is called `uncertainty-aware-ntfield`.

Start the Docker container:

```bash
docker compose up -d
```

Enter the running container:

```bash
docker exec -it <container-name> bash
```

## Build Torch-KDTree from Source

Inside the container, build and install `torch-kdtree` from source.

First, initialize the submodules:

```bash
cd /workspace/torch_kdtree
git submodule update --init --recursive
```

The submodule command is important because `torch-kdtree` depends on `pybind11`. If `pybind11` is missing, CMake may fail with an error saying that the `pybind11` folder does not contain a `CMakeLists.txt` file.

### Set the CUDA Architecture List

Before running `pip install .`, open `CMakeLists.txt` and, around line 106, set the CUDA architecture list to match your GPU. For example, for newer NVIDIA GPUs, we used:

```cmake
set(CUDA_ARCH_LIST "75-real;80-real;86-real;89-real;90-virtual" CACHE STRING
    "Semicolon-separated CUDA architecture list (SM >= 70 required for CUDA 13+)")
```

Users may need to modify this line depending on their own GPU and CUDA version.

Example architecture values:

```cmake
75-real      # Turing, e.g. RTX 20-series
80-real      # Ampere, e.g. A100
86-real      # Ampere, e.g. RTX 30-series
89-real      # Ada, e.g. RTX 40-series
90-virtual   # Hopper / forward-compatible virtual architecture
```

### Install

Once the architecture list is set, install the package:

```bash
pip install .
```

If the package was previously installed incorrectly, reinstall it with:

```bash
pip uninstall torch-kdtree -y
pip install .
```

## Run the Training Pipeline

After the container is running and `torch-kdtree` has been built successfully:

Sample the training data:

```bash
python dataprocessing/preprocess.py --config configs/gibson.txt
```

Start the training:

```bash
python train/train_gib.py
```

## Expected Workflow Summary

```bash
cd .devcontainer
# Edit docker-compose.yml line 19: set the path before ':' to this folder's path
docker compose up -d
docker exec -it <container-name> bash

cd /workspace/torch_kdtree
git submodule update --init --recursive
# Edit CMakeLists.txt (~line 106) to set CUDA_ARCH_LIST for your GPU
pip install .

# From the repository root inside the container:
python dataprocessing/preprocess.py --config configs/gibson.txt
python train/train_gib.py
```

## Notes

- `torch-kdtree` should be built inside the Docker container, not on the host machine.
- If CMake fails during the `torch-kdtree` build, first check that the `pybind11` submodule was initialized correctly.
- If CUDA-related errors occur, verify that the container has GPU access by running:

```bash
nvidia-smi
```

inside the container.


```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64
export CPATH=$CUDA_HOME/include
```

## Generate and Optimize Nominal Paths

After training, generate nominal paths and run the trajectory optimization:

1. Generate nominal paths with `tests/arm_plan_rand_with_time.py`. The output is saved to `Evaluations/Arm/paths_ours_speed_up`.

```bash
   python tests/arm_plan_rand_with_time.py
```

2. Copy the `ours_speed_up` directory into the optimizer folder:

```bash
   cp -r Evaluations/Arm/ours_speed_up \
     uncertainty-aware-active-ntfields/manipulator/optimize_manipulator_trajectory/
```

3. Run the optimizer from that directory:

```bash
   cd uncertainty-aware-active-ntfields/manipulator/optimize_manipulator_trajectory
   python optimize_all_trajectories.py
```

4. Copy `optimized_paths_working_variable_sigma` back into `Evaluations/Arm/`:

```bash
   cp -r Evaluations/Arm/optimized_paths_working_variable_sigma ./Evaluations/Arm/
```

5. Check for collisions:

```bash
   python check_optimized_path.py
```

6. Compare path lengths:

```bash
   python compare_path_lengths_all_with_ours_optimized.py
```
