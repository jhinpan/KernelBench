# Docker-based SGLang Server with Separated Generation and Evaluation

This document explains how to use Docker for the SGLang server (generation) and run evaluation locally.

## Overview

The workflow separates kernel generation (using Docker SGLang server) from evaluation (run locally with GPU):

1. **Generation**: Uses Docker container running SGLang server
2. **Evaluation**: Runs locally with GPU access

## Setup

### 1. Start SGLang Server in Docker

```bash
# Example Docker command (adjust based on your Docker image)
docker run -d \
  --name sglang-server \
  -p 30000:30000 \
  --gpus all \
  lmsysorg/sglang:latest \
  python -m sglang \
    --model-path deepseek-ai/deepseek-coder-6.7b-instruct \
    --port 30000 \
    --host 0.0.0.0
```

### 2. Verify Server is Running

```bash
# Check if server is accessible
curl http://localhost:30000/health
```

## Workflow

### Step 1: Generate Kernel (Using Docker SGLang)

```bash
python3 scripts/generate_single_sample.py \
  run_name="my_experiment" \
  dataset_src="huggingface" \
  level=2 \
  problem_id=40 \
  server_type="sglang" \
  model_name="deepseek-ai/deepseek-coder-6.7b-instruct"
```

This will:
- Connect to the SGLang server running in Docker on port 30000
- Generate a CUDA kernel for the specified problem
- Save outputs to `runs/my_experiment/level_2/problem_40/`
  - `kernel.py`: Generated CUDA kernel
  - `reference.py`: Original PyTorch code
  - `metadata.json`: Generation metadata
  - `raw_output.txt`: Raw model output
  - `prompt.txt`: (if log_prompt=True)

### Step 2: Evaluate Kernel (Locally with GPU)

```bash
python3 scripts/eval_single_sample.py \
  run_name="my_experiment" \
  dataset_src="huggingface" \
  level=2 \
  problem_id=40 \
  num_correct_trials=5 \
  num_perf_trials=100
```

This will:
- Load the pre-generated kernel from `runs/my_experiment/level_2/problem_40/kernel.py`
- Evaluate correctness and performance against the reference
- Save results to `runs/my_experiment/level_2/problem_40/eval_result.json`

## Batch Processing

For multiple problems, you can use the existing batch scripts:

### Batch Generation
```bash
python3 scripts/generate_samples.py \
  run_name="batch_experiment" \
  dataset_src="huggingface" \
  level=1 \
  server_type="sglang" \
  model_name="deepseek-ai/deepseek-coder-6.7b-instruct" \
  num_workers=10
```

### Batch Evaluation
```bash
python3 scripts/eval_from_generations.py \
  run_name="batch_experiment" \
  dataset_src="huggingface" \
  level=1 \
  num_gpu_devices=1 \
  timeout=300
```

## Troubleshooting

### Connection Issues
If the generation script can't connect to the Docker SGLang server:

1. Check Docker container is running: `docker ps`
2. Check port mapping: `docker port sglang-server`
3. Check firewall settings
4. Try using Docker's host IP instead of localhost

### Custom Server Configuration
If your SGLang server has a different configuration, you can modify the inference server creation in the generation script or pass custom parameters.

## Benefits of This Approach

1. **Isolation**: Generation server runs in controlled Docker environment
2. **Scalability**: Can run multiple Docker containers for parallel generation
3. **Flexibility**: Can use different models/configs without affecting evaluation environment
4. **Resource Management**: Generation and evaluation can use different hardware resources