import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.utils import set_gpu_arch, read_file

"""
Evaluate a single pre-generated kernel sample

This script evaluates a kernel that was previously generated and saved to disk.
It reads the kernel from runs/{run_name}/level_{level}/problem_{problem_id}/kernel.py

Usage:
```
python3 scripts/eval_single_sample.py run_name="my_run" dataset_src="huggingface" level=2 problem_id=40
```
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local
        
        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Run name to find the generated kernel
        self.run_name = REQUIRED

        # Evaluation settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.measure_performance = True
        
        # GPU architecture
        self.gpu_arch = ["Hopper"]

        # Directories
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        
        # Logging
        self.verbose = False
        self.save_results = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Evaluate a single pre-generated kernel sample
    """
    print(f"Starting Evaluation with config: {config}")

    # Set GPU architecture
    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)

    # Load dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Starting Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Reference Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

    # 2. Load Generated Kernel
    kernel_dir = os.path.join(config.runs_dir, config.run_name, f"level_{config.level}", f"problem_{config.problem_id}")
    kernel_path = os.path.join(kernel_dir, "kernel.py")
    
    if not os.path.exists(kernel_path):
        print(f"ERROR: Kernel file not found at {kernel_path}")
        print("Make sure you've run the generation script first.")
        return
    
    custom_cuda = read_file(kernel_path)
    
    # Check if generation failed
    if custom_cuda.startswith("# GENERATION FAILED"):
        print("ERROR: The kernel generation failed. Cannot evaluate.")
        return

    # 3. Evaluate Kernel
    print(f"Evaluating kernel from: {kernel_path}")
    print(f"Correctness trials: {config.num_correct_trials}")
    print(f"Performance trials: {config.num_perf_trials}")
    
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, 
        custom_cuda, 
        verbose=config.verbose, 
        measure_performance=config.measure_performance, 
        num_correct_trials=config.num_correct_trials, 
        num_perf_trials=config.num_perf_trials
    )
    
    print(f"\nEvaluation result for level {config.level} problem {config.problem_id}:")
    print(f"Problem Name: {problem_name}")
    print(kernel_exec_result)

    # 4. Save Results
    if config.save_results:
        result_dict = {
            "problem_id": config.problem_id,
            "problem_name": problem_name,
            "level": config.level,
            "success": kernel_exec_result.success,
            "correctness": kernel_exec_result.correctness,
            "runtime_ms": kernel_exec_result.runtime_ms,
            "ref_runtime_ms": kernel_exec_result.ref_runtime_ms,
            "speedup": kernel_exec_result.speedup,
            "num_correct_trials": config.num_correct_trials,
            "num_perf_trials": config.num_perf_trials,
            "error": kernel_exec_result.error
        }
        
        # Add metadata if available
        metadata_path = os.path.join(kernel_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                result_dict["generation_metadata"] = metadata
        
        # Save evaluation results
        result_path = os.path.join(kernel_dir, "eval_result.json")
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n✓ Evaluation results saved to: {result_path}")


if __name__ == "__main__":
    main()