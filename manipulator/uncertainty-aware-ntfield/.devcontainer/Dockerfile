# Start from the iGibson base image or another if more appropriate
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
#FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pickle5
RUN pip install libigl
RUN pip install pytorch_kinematics
RUN pip install configargparse
#RUN pip install pointnext
RUN pip install functorch==0.2.1 pytorch_kinematics==0.6



RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libxfixes3 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libgl1-mesa-dri
    && rm -rf /var/lib/apt/lists/*
RUN pip install trimesh pyglet==1.5.16 mujoco
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$CUDA_HOME/lib64
ENV CPATH=$CUDA_HOME/include

WORKDIR /workspace
