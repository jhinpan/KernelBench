#!/usr/bin/env python3
"""
Generate baseline timing results for H100_atlas setup
This script profiles the wall clock time for each KernelBench reference problem
on the H100_atlas GPU configuration.

Configurations tested:
- PyTorch Eager execution
- Torch Compile with inductor backend (default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs)
- Torch Compile with cudagraphs backend
"""

import torch
import numpy as np
from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
    fetch_ref_arch_from_problem_id,
)
from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file
import os
import json
from tqdm import tqdm
import sys

# Add the KernelBench directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")

# Hardware configuration name
HARDWARE_NAME = "H100_atlas"


def fetch_ref_arch_from_dataset(dataset: list[str], 
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str, 
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor", 
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            if use_torch_compile:
                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")
            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")
            
            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")
        return None


def record_baseline_times(use_torch_compile: bool = False, 
                          torch_compile_backend: str="inductor", 
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json"):
    """
    Generate baseline time for KernelBench, 
    configure profiler options for PyTorch
    save to specified file
    """
    device = torch.device("cuda:0")
    json_results = {}
    
    for level in [1, 2, 3]:
        PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
        dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
        json_results[f"level{level}"] = {}

        num_problems = len(dataset)
        print(f"\nProcessing Level {level} with {num_problems} problems...")
        
        for problem_id in tqdm(range(1, num_problems + 1), desc=f"Level {level}"):
            ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
            runtime_stats = measure_program_time(
                ref_arch_name=ref_arch_name,
                ref_arch_src=ref_arch_src,
                use_torch_compile=use_torch_compile,
                torch_compile_backend=torch_compile_backend,
                torch_compile_options=torch_compile_options,
                device=device,
                verbose=False # do not print 
            )
            if runtime_stats is not None:
                json_results[f"level{level}"][ref_arch_name] = runtime_stats

    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Saved results to: {save_path}")
    return json_results


def main():
    """
    Main function to generate all baseline timings for H100_atlas
    """
    print(f"Starting baseline timing generation for {HARDWARE_NAME}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print("-" * 80)
    
    # Check if directory already exists
    if os.path.exists(os.path.join(TIMING_DIR, HARDWARE_NAME)):
        response = input(f"\nDirectory {HARDWARE_NAME} already exists. Do you want to overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting...")
            return

    print("\n" + "="*80)
    print("PHASE 1: PyTorch Eager Execution")
    print("="*80)
    record_baseline_times(
        use_torch_compile=False, 
        torch_compile_backend=None,
        torch_compile_options=None, 
        file_name=f"{HARDWARE_NAME}/baseline_time_torch.json"
    )
    
    print("\n" + "="*80)
    print("PHASE 2: Torch Compile with Inductor Backend")
    print("="*80)
    for torch_compile_mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
        print(f"\nMode: {torch_compile_mode}")
        print("-" * 40)
        record_baseline_times(
            use_torch_compile=True, 
            torch_compile_backend="inductor",
            torch_compile_options=torch_compile_mode, 
            file_name=f"{HARDWARE_NAME}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json"
        )

    print("\n" + "="*80)
    print("PHASE 3: Torch Compile with CUDAGraphs Backend")
    print("="*80)
    record_baseline_times(
        use_torch_compile=True, 
        torch_compile_backend="cudagraphs",
        torch_compile_options=None, 
        file_name=f"{HARDWARE_NAME}/baseline_time_torch_compile_cudagraphs.json"
    )
    
    print("\n" + "="*80)
    print(f"Baseline timing generation complete for {HARDWARE_NAME}!")
    print(f"Results saved in: {os.path.join(TIMING_DIR, HARDWARE_NAME)}")
    print("="*80)


if __name__ == "__main__":
    main()