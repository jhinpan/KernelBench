# Use NVIDIA CUDA base image with Python support
# This includes CUDA 12.1, cuDNN, and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /workspace/KernelBench

# Copy requirements.txt first to leverage Docker layer cache
COPY requirements.txt .

# Create conda environment and install dependencies
RUN conda create -n kernel-bench python=3.10 -y && \
    echo "source activate kernel-bench" >> ~/.bashrc

# Activate environment and install PyTorch with CUDA support
# Note: Installing PyTorch separately to ensure CUDA version compatibility
RUN /bin/bash -c "source activate kernel-bench && \
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt"

# Copy the entire project
COPY . .

# Install the project in the conda environment
RUN /bin/bash -c "source activate kernel-bench && pip install -e ."

# Set environment variable to ensure correct Python is used
ENV PATH="/opt/conda/envs/kernel-bench/bin:$PATH"

# Create directories for runs and cache
RUN mkdir -p runs cache

# Set up environment for interactive use
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Default to activating the conda environment
ENTRYPOINT ["/bin/bash", "-c", "source activate kernel-bench && exec \"$@\"", "--"]
CMD ["/bin/bash"]