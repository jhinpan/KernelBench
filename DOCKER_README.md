# Docker Setup for KernelBench

This guide explains how to use Docker to run KernelBench with full CUDA support.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Docker runtime** installed:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Docker** and **Docker Compose** installed

## Quick Start

### Option 1: Use Pre-built Image from Docker Hub

```bash
# Pull the latest image
docker pull jhinpan/kernelbenchimage:latest

# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/KernelBench \
  jhinpan/kernelbenchimage:latest bash
```

### Option 2: Build Locally

```bash
# Build using docker-compose
docker-compose build

# Or build directly with Docker
docker build -t jhinpan/kernelbenchimage:latest .
```

### 2. Run the Container

#### Using Docker Compose (Recommended)

```bash
# Start interactive container
docker-compose run --rm kernelbench bash

# Run a specific command
docker-compose run --rm kernelbench python scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=1 problem_id=1
```

#### Using Docker Directly

```bash
# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/KernelBench \
  -e NVIDIA_VISIBLE_DEVICES=all \
  kernelbench:latest bash
```

### 3. Set API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
TOGETHER_API_KEY=your_together_key
```

Then uncomment the environment variables in `docker-compose.yml`.

## Common Commands in Docker

Once inside the container:

```bash
# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

# Run single problem evaluation
python scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=1 problem_id=1

# Generate baseline timings
python scripts/generate_baseline_time.py

# Run full benchmark
python scripts/generate_samples.py run_name=docker_test dataset_src=huggingface level=1 num_workers=10
python scripts/eval_from_generations.py run_name=docker_test dataset_src=local level=1
```

## Features of This Docker Setup

1. **CUDA Support**: Based on `nvidia/cuda:12.1.0` with cuDNN for full GPU acceleration
2. **Persistent Storage**: Uses Docker volumes for `cache/` and `runs/` directories
3. **Development Mode**: Mounts project directory for live code changes
4. **Pre-configured Environment**: Conda environment with all dependencies installed
5. **GPU Access**: Automatic GPU detection and configuration

## Troubleshooting

### CUDA Not Found

If CUDA is not detected:

1. Ensure NVIDIA drivers are installed on host:
   ```bash
   nvidia-smi
   ```

2. Check Docker GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. Verify NVIDIA Container Runtime:
   ```bash
   docker info | grep nvidia
   ```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run container with your user ID
docker-compose run --rm --user $(id -u):$(id -g) kernelbench bash
```

### Building Issues

If the build fails:

1. Clean Docker cache:
   ```bash
   docker system prune -a
   ```

2. Build with no cache:
   ```bash
   docker-compose build --no-cache
   ```

## Advanced Usage

### Using Multiple GPUs

```bash
# Use specific GPUs
docker run --gpus '"device=0,1"' -it --rm kernelbench:latest bash

# In docker-compose.yml, modify:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']
          capabilities: [gpu]
```

### Custom CUDA Version

To use a different CUDA version, modify the base image in Dockerfile:

```dockerfile
# For CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Update PyTorch installation accordingly
RUN pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

## Performance Tips

1. **Build Cache**: The Dockerfile is optimized to cache dependency installation
2. **Volume Mounts**: Results are stored in Docker volumes for persistence
3. **Network Mode**: Uses host network for better API performance
4. **GPU Memory**: Container has full access to GPU memory

## Building and Publishing to Docker Hub

### Build and Push Using Script

We provide a convenient script to build and push images:

```bash
# Build and push with latest tag
./docker-build-push.sh

# Build and push with specific tag
./docker-build-push.sh v1.0.0

# Push only (skip build)
./docker-build-push.sh --push-only v1.0.0
```

### Manual Build and Push

```bash
# 1. Login to Docker Hub
docker login

# 2. Build the image
docker-compose build
# Or: docker build -t jhinpan/kernelbenchimage:latest .

# 3. Push to Docker Hub
docker push jhinpan/kernelbenchimage:latest

# 4. (Optional) Tag and push with version
docker tag jhinpan/kernelbenchimage:latest jhinpan/kernelbenchimage:v1.0.0
docker push jhinpan/kernelbenchimage:v1.0.0
```

### Multi-Architecture Build (Optional)

For building images that work on multiple architectures (AMD64, ARM64):

```bash
# Create and use buildx builder
docker buildx create --name multiarch --use

# Build and push multi-arch image
docker buildx build --platform linux/amd64,linux/arm64 \
  -t jhinpan/kernelbenchimage:latest \
  --push .
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes cached data)
docker-compose down -v

# Remove local images
docker rmi jhinpan/kernelbenchimage:latest
```