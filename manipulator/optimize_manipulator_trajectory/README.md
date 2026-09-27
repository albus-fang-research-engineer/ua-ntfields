# Manipulator Trajectory Optimization Demo

This repository contains the manipulator trajectory optimization demo for the NTRL / Active Neural Time Field pipeline.

## Docker Setup

From the repository root, enter the devcontainer folder:

```bash
cd optimize_manipulator_trajectory/.devcontainer
```
Before starting the container, open `.devcontainer/docker-compose.yml` and, on **line 19**, change the directory before the `:` to the actual path of this folder on your machine. This folder is called `optimize_manipulator_trajectory`.

Start the Docker container:

```bash
docker compose up -d
```

Enter the running container:

```bash
docker exec -it ntrl-demo-manipulator-demo bash
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

## Run the Manipulator Optimization

After the container is running and `torch-kdtree` has been built successfully, run:

```bash
cd /workspace/optimize_manipulator_trajectory
python3 optimize_all_trajectories.py
```

This script runs the full manipulator trajectory optimization pipeline.

## Expected Workflow Summary

```bash
cd optimize_manipulator_trajectory/.devcontainer
docker compose up -d
docker exec -it ntrl-demo-manipulator-demo bash

cd /workspace/torch_kdtree
git submodule update --init --recursive
# Edit CMakeLists.txt (~line 106) to set CUDA_ARCH_LIST for your GPU
pip install .

cd /workspace/optimize_manipulator_trajectory
python3 optimize_all_trajectories.py
```

## Notes

- The container name is:

```bash
ntrl-demo-manipulator-demo
```

- `torch-kdtree` should be built inside the Docker container, not on the host machine.
- If CMake fails during the `torch-kdtree` build, first check that the `pybind11` submodule was initialized correctly.
- If CUDA-related errors occur, verify that the container has GPU access by running:

```bash
nvidia-smi
```

inside the container.