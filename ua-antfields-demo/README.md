# UA-ANTFields Demo

This repository contains the demo for **Uncertainty-Aware Active Neural Time Fields (UA-ANTFields)**.

## Docker Setup

From the repository root, enter the devcontainer folder:

```bash
cd .devcontainer
```

Before starting the container, open `.devcontainer/docker-compose.yml` and, on **line 14**, change the directory before the `:` to the actual path of this folder on your machine. This folder is called `ua-antfields-demo`.

Start the Docker container:

```bash
docker compose up -d
```

Enter the running container:

```bash
docker exec -it <container-name> bash
```

## Running the Demo

```bash
python main.py
```

This runs the selected start and goal pair in the Gibson environment **Superior** 10 times.

- To change the start and goal, open `models/model_gibson.py` and edit `self.fixed_goal` and `initial_view`.
- To change the mesh, open `main.py` and edit `meshpath`.

## Notes

- If CUDA-related errors occur, verify that the container has GPU access by running:

```bash
nvidia-smi
```

inside the container.