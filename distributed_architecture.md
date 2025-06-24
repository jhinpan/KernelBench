# Distributed KernelBench Architecture

This document describes the distributed architecture for separating kernel generation (rollout) and evaluation across different Docker containers.

## Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   RL Framework      │     │   SGLang Servers    │     │   KernelBench       │
│     Docker          │────▶│     (Rollout)       │────▶│   Eval Docker       │
│                     │     │                     │     │                     │
│ - RL Training Loop  │     │ - Model Inference   │     │ - Kernel Evaluation │
│ - Orchestration     │     │ - Code Generation   │     │ - Performance Test  │
│                     │     │                     │     │ - Correctness Check │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                          Communication via HTTP APIs
```

## Components

### 1. RL Framework Docker
- Manages the reinforcement learning training loop
- Orchestrates rollouts and evaluations
- Collects rewards from evaluations
- Updates model parameters

### 2. SGLang Servers
- Handle model inference for code generation
- Can scale horizontally for parallel rollouts
- Generate CUDA kernels from PyTorch reference code

### 3. KernelBench Evaluation Docker
- Dedicated container for kernel evaluation
- Provides REST API for evaluation requests
- Returns performance metrics and correctness results

## Setup Instructions

### 1. Build KernelBench Evaluation Docker

Create a Dockerfile for the evaluation service:

```dockerfile
# Dockerfile.kernelbench-eval
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flask

# Copy KernelBench code
COPY . /workspace/KernelBench

# Set working directory
WORKDIR /workspace/KernelBench

# Expose evaluation service port
EXPOSE 8080

# Start evaluation service
CMD ["python", "scripts/kernelbench_eval_service.py", "--port", "8080"]
```

Build and run:
```bash
docker build -f Dockerfile.kernelbench-eval -t kernelbench-eval:latest .
docker run -d --name kernelbench-eval --gpus all -p 8080:8080 kernelbench-eval:latest
```

### 2. Start SGLang Servers

```bash
# Start multiple SGLang servers for parallel rollouts
docker run -d --name sglang-1 --gpus '"device=0"' -p 30001:30000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path deepseek-ai/deepseek-coder-6.7b-instruct \
    --port 30000

docker run -d --name sglang-2 --gpus '"device=1"' -p 30002:30000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path deepseek-ai/deepseek-coder-6.7b-instruct \
    --port 30000
```

### 3. RL Framework Integration

In your RL framework Docker, use the provided client library:

```python
from src.eval_client import KernelBenchEvalClient
from your_rl_framework import RLTrainer, SGLangRolloutManager

# Initialize components
eval_client = KernelBenchEvalClient("http://kernelbench-eval:8080")
rollout_manager = SGLangRolloutManager(
    servers=["http://sglang-1:30000", "http://sglang-2:30000"]
)

# RL training loop
trainer = RLTrainer()

for episode in range(num_episodes):
    # 1. Generate rollouts using SGLang servers
    rollouts = rollout_manager.generate_rollouts(
        problems=trainer.get_current_problems(),
        policy=trainer.policy
    )
    
    # 2. Evaluate generated kernels
    eval_requests = [
        {
            "dataset_src": "huggingface",
            "level": rollout["level"],
            "problem_id": rollout["problem_id"],
            "kernel_code": rollout["generated_code"]
        }
        for rollout in rollouts
    ]
    
    eval_results = eval_client.batch_evaluate(eval_requests)
    
    # 3. Calculate rewards from evaluation results
    rewards = []
    for result in eval_results:
        if result["success"] and result["correctness"]:
            # Reward based on speedup
            reward = max(0, result["speedup"] - 1.0)
        else:
            # Penalty for incorrect or failed kernels
            reward = -1.0
        rewards.append(reward)
    
    # 4. Update policy
    trainer.update_policy(rollouts, rewards)
```

## API Reference

### KernelBench Evaluation Service

#### Health Check
```
GET /health
```

Response:
```json
{
    "status": "healthy",
    "service": "kernelbench_eval",
    "cuda_available": true,
    "gpu_count": 1
}
```

#### Evaluate Single Kernel
```
POST /evaluate
```

Request:
```json
{
    "dataset_src": "huggingface",
    "dataset_name": "ScalingIntelligence/KernelBench",
    "level": 2,
    "problem_id": 40,
    "kernel_code": "...",
    "num_correct_trials": 5,
    "num_perf_trials": 100,
    "gpu_arch": ["Hopper"],
    "verbose": false
}
```

Response:
```json
{
    "success": true,
    "problem_name": "40_Conv2d_Bias_ReLU.py",
    "correctness": true,
    "runtime_ms": 0.123,
    "ref_runtime_ms": 0.456,
    "speedup": 3.71,
    "error": null
}
```

#### Batch Evaluate
```
POST /batch_evaluate
```

Request:
```json
{
    "evaluations": [
        {
            "dataset_src": "huggingface",
            "level": 1,
            "problem_id": 1,
            "kernel_code": "..."
        },
        ...
    ],
    "common_params": {
        "num_correct_trials": 5,
        "num_perf_trials": 100
    }
}
```

## Docker Compose Example

```yaml
version: '3.8'

services:
  kernelbench-eval:
    build:
      context: .
      dockerfile: Dockerfile.kernelbench-eval
    container_name: kernelbench-eval
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - rl-network

  sglang-1:
    image: lmsysorg/sglang:latest
    container_name: sglang-1
    ports:
      - "30001:30000"
    command: >
      python -m sglang.launch_server
      --model-path deepseek-ai/deepseek-coder-6.7b-instruct
      --port 30000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    networks:
      - rl-network

  rl-framework:
    build:
      context: ./rl-framework
      dockerfile: Dockerfile
    container_name: rl-framework
    environment:
      - KERNELBENCH_URL=http://kernelbench-eval:8080
      - SGLANG_URLS=http://sglang-1:30000
    depends_on:
      - kernelbench-eval
      - sglang-1
    networks:
      - rl-network

networks:
  rl-network:
    driver: bridge
```

## Performance Considerations

1. **Batch Evaluation**: Use batch evaluation endpoint to reduce overhead
2. **Connection Pooling**: The client library uses session pooling
3. **Timeouts**: Adjust timeouts based on problem complexity
4. **GPU Allocation**: Ensure proper GPU allocation across services
5. **Caching**: The evaluation service caches datasets to reduce loading time

## Monitoring

Monitor service health and performance:

```bash
# Check service status
curl http://localhost:8080/health

# Monitor Docker containers
docker stats kernelbench-eval sglang-1 sglang-2

# View logs
docker logs -f kernelbench-eval
```