# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

KernelBench is a benchmark for evaluating Large Language Models' ability to generate efficient GPU kernels. It provides a systematic framework to test LLMs on transpiling PyTorch operators to CUDA kernels across four difficulty levels.

## Common Development Commands

### Setup
```bash
# Create environment
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -r requirements.txt
pip install -e .

# Set API keys for LLM providers
export {PROVIDER}_API_KEY=your_api_key  # PROVIDER: OPENAI, ANTHROPIC, GOOGLE, TOGETHER, etc.

# For Modal (remote GPU) setup
modal token new
```

### Testing
```bash
# Run all unit tests
pytest src/unit_tests/

# Run specific test file
pytest src/unit_tests/test_dataset.py
```

### Evaluating Single Kernels
```bash
# Generate and evaluate a single problem
python3 scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=2 problem_id=40

# Run with Modal (for remote GPU)
python3 scripts/generate_and_eval_single_sample_modal.py dataset_src="huggingface" level=2 problem_id=40

# Check a generated kernel against reference
python3 scripts/run_and_check.py ref_origin=kernelbench level=1 problem_id=5 kernel_src_path=path/to/kernel.py
```

### Full Benchmark Workflow
```bash
# 1. Generate kernels from LLM
python3 scripts/generate_samples.py run_name=my_run dataset_src=huggingface level=1 num_workers=50 server_type=openai model_name=gpt-4

# 2. Evaluate generated kernels
python3 scripts/eval_from_generations.py run_name=my_run dataset_src=local level=1 num_gpu_devices=8 timeout=300

# 3. Analyze results
python3 scripts/benchmark_eval_analysis.py run_name=my_run level=1 hardware=L40S_matx3 baseline=baseline_time_torch
```

## Code Architecture

### Directory Structure
- `KernelBench/`: Problem dataset organized by levels
  - `level1/`: Single-kernel operators (convolutions, matmuls, activations)
  - `level2/`: Fusion patterns (conv+bias+relu, matmul+scale+sigmoid)
  - `level3/`: Full architectures (MobileNet, VGG, MiniGPT, Mamba)
  - `level4/`: HuggingFace models

- `src/`: Core evaluation infrastructure
  - `dataset.py`: Problem loading and management
  - `eval.py`: Kernel evaluation (correctness & performance)
  - `compile.py`: Parallel compilation infrastructure
  - `score.py`: Metrics calculation (speedup, Fast@p)
  - `prompt_constructor.py`: LLM prompt generation
  - `utils.py`: LLM APIs and utilities

### Problem Format
Each problem must define:
```python
class Model(nn.Module):
    def __init__(self, *args):
        # Model initialization
    
    def forward(self, *inputs):
        # PyTorch operations to optimize

def get_inputs():
    # Return representative input tensors

def get_init_inputs():
    # Return model initialization parameters
```

### Evaluation Pipeline
1. LLM generates CUDA kernel from PyTorch model
2. Kernel is compiled (optionally pre-compiled on CPU)
3. Correctness check: compare outputs with PyTorch reference
4. Performance measurement: time kernel vs PyTorch baseline
5. Score calculation: Fast@p metric (fraction both correct and p× faster)

### Key Design Patterns
- Problems are identified by content hash for reproducibility
- Evaluation uses multiple random seeds for robustness
- GPU cleanup happens automatically after each evaluation
- Results are cached to avoid redundant computation
- Parallel processing for both compilation and evaluation

## Important Notes

- KernelBench uses PyTorch 2.5.0 with CUDA support
- Ninja build system is used for faster kernel compilation
- No linting tools are configured - follow existing code style
- Tests require GPU access for full functionality
- Modal can be used for remote GPU execution
- Baseline timings are hardware-specific - regenerate for accurate comparisons