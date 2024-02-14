# Start from the NVIDIA PyTorch image with CUDA 11.1
FROM nvcr.io/nvidia/pytorch:21.02-py3

# Set the working directory
WORKDIR /workspace

# Set up Python 3.8
RUN apt-get update && apt-get install -y python3.8 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install OpenGL libraries required for Open3D
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev

# Install PyTorch, torchvision, and torchaudio
# Note: Verify if these installations are necessary as they might already be included in the base image.
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
RUN pip install fvcore iopath \
    && pip install 'git+https://github.com/facebookresearch/fvcore' \
    && pip install 'git+https://github.com/facebookresearch/iopath'

# Install PyTorch3D
RUN pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2'

# Install additional project dependencies
RUN pip install open3d timm easydict transforms3d h5py einops tensorboardX wandb opencv-python-headless

# Copy the project files into the container
COPY . /workspace

# Keep the container running indefinitely (tail -f /dev/null is a "do nothing" command that runs indefinitely)
ENTRYPOINT ["tail", "-f", "/dev/null"]