#!/usr/bin/env python3
"""
KernelBench Full Pipeline with Qwen3-235B-A22B (localhost:30000)

This script provides a Python interface to run the complete KernelBench pipeline:
1. Generate kernel implementations using the model at localhost:30000
2. Evaluate the generated kernels (correctness + performance)
3. Analyze results and compute metrics

Usage:
    # Basic usage with defaults (Level 1, 4 workers)
    python3 scripts/run_kernelbench_with_qwen3_235b.py

    # Custom configuration
    python3 scripts/run_kernelbench_with_qwen3_235b.py \
        --level 2 \
        --num-workers 8 \
        --run-name my_custom_run \
        --num-gpu-devices 2

    # Run only specific steps
    python3 scripts/run_kernelbench_with_qwen3_235b.py \
        --run-name existing_run \
        --skip-generation \
        --skip-analysis

    # Test with single problem first
    python3 scripts/run_kernelbench_with_qwen3_235b.py \
        --level 1 \
        --subset 0 1 \
        --run-name test_run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print a formatted header"""
    print()
    print("=" * 80)
    print(f"{Colors.BOLD}{message}{Colors.ENDC}")
    print("=" * 80)
    print()


def print_info(message):
    """Print an info message"""
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {message}")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}", file=sys.stderr)


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")


def check_server(server_address, server_port):
    """Check if the model server is running and responding"""
    print_info(f"Checking if model server is running at {server_address}:{server_port}...")

    endpoints_to_try = [
        f"http://{server_address}:{server_port}/health",
        f"http://{server_address}:{server_port}/v1/models",
    ]

    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print_success("✓ Server is responding")
                return True
        except requests.exceptions.RequestException:
            continue

    print_error(f"✗ Server is not responding at http://{server_address}:{server_port}")
    print_error("Please ensure your model server is running before proceeding.")
    print_error("You can start it with: ./launch_qwen3_235b_a22b_server.sh")
    return False


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print_info(f"{description}...")
    print_info(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=KERNELBENCH_DIR)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print_error(f"{description} failed with return code {result.returncode}")
        return False

    print_success(f"✓ {description} completed in {elapsed_time:.2f}s")
    return True


def generate_samples(args):
    """Step 1: Generate kernel samples using the model"""
    print_header("Step 1/3: Generating Kernel Samples")

    cmd = [
        "python3", "scripts/generate_samples.py",
        f"run_name={args.run_name}",
        f"dataset_src={args.dataset_src}",
        f"level={args.level}",
        f"server_type={args.server_type}",
        f"server_address={args.server_address}",
        f"server_port={args.server_port}",
        f"model_name={args.model_name}",
        f"num_workers={args.num_workers}",
        f"num_samples={args.num_samples}",
    ]

    # Add subset if specified
    if args.subset:
        cmd.append(f"subset=({args.subset[0]},{args.subset[1]})")

    print_info(f"Starting generation with {args.num_workers} workers...")
    print_info("This may take a while depending on the number of problems and model speed...")

    if not run_command(cmd, "Generation"):
        return False

    print_success(f"Generated samples stored in: runs/{args.run_name}/")
    return True


def evaluate_samples(args):
    """Step 2: Evaluate the generated kernels"""
    print_header("Step 2/3: Evaluating Generated Kernels")

    cmd = [
        "python3", "scripts/eval_from_generations.py",
        f"run_name={args.run_name}",
        f"dataset_src={args.dataset_src}",
        f"level={args.level}",
        f"num_gpu_devices={args.num_gpu_devices}",
        f"n_correct={args.num_trials_correctness}",
        f"n_trials={args.num_trials_performance}",
    ]

    print_info(f"Starting evaluation with {args.num_gpu_devices} GPU devices...")
    print_info(f"Correctness trials: {args.num_trials_correctness}")
    print_info(f"Performance trials: {args.num_trials_performance}")

    if not run_command(cmd, "Evaluation"):
        return False

    print_success(f"Evaluation results stored in: runs/{args.run_name}/eval_results.json")
    return True


def analyze_results(args):
    """Step 3: Analyze the benchmark results"""
    print_header("Step 3/3: Analyzing Benchmark Results")

    cmd = [
        "python3", "scripts/benchmark_eval_analysis.py",
        f"run_name={args.run_name}",
        f"level={args.level}",
        f"hardware={args.hardware}",
        f"baseline={args.baseline}",
    ]

    print_info("Computing metrics and performance analysis...")

    if not run_command(cmd, "Analysis"):
        return False

    print_success("Analysis completed")
    return True


def print_summary(args):
    """Print a summary of the pipeline execution"""
    print_header("Pipeline Completed Successfully!")

    run_dir = KERNELBENCH_DIR / "runs" / args.run_name

    print(f"Results Summary:")
    print(f"  Run Name:     {args.run_name}")
    print(f"  Location:     {run_dir}")
    print(f"  Generations:  {run_dir / 'generations'}")
    print(f"  Evaluations:  {run_dir / 'eval_results.json'}")
    print()
    print("To view detailed results:")
    print(f"  cat {run_dir / 'eval_results.json'}")
    print()
    print("To inspect specific problem generations:")
    print(f"  ls {run_dir / 'generations'}")
    print()
    print_success("All done! 🎉")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run KernelBench pipeline with Qwen3-235B-A22B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model Server Configuration
    parser.add_argument("--server-type", default="local",
                        help="Server type (default: local)")
    parser.add_argument("--server-address", default="localhost",
                        help="Server address (default: localhost)")
    parser.add_argument("--server-port", type=int, default=30000,
                        help="Server port (default: 30000)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-235B-A22B",
                        help="Model name (default: Qwen/Qwen3-235B-A22B)")

    # Run Configuration
    parser.add_argument("--run-name", default=None,
                        help=f"Name for this run (default: qwen3_235b_run_<timestamp>)")
    parser.add_argument("--dataset-src", default="local", choices=["local", "huggingface"],
                        help="Dataset source (default: local)")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Difficulty level 1-4 (default: 1)")
    parser.add_argument("--subset", type=int, nargs=2, metavar=("START", "END"),
                        help="Subset of problems to run (start, end)")

    # Parallel Configuration
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for generation (default: 4)")
    parser.add_argument("--num-gpu-devices", type=int, default=2,
                        help="Number of GPU devices for evaluation (default: 2)")

    # Evaluation Configuration
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per problem (default: 1)")
    parser.add_argument("--num-trials-correctness", type=int, default=5,
                        help="Number of correctness trials (default: 5)")
    parser.add_argument("--num-trials-performance", type=int, default=100,
                        help="Number of performance trials (default: 100)")

    # Hardware Configuration
    parser.add_argument("--hardware", default="H200",
                        help="Hardware type for benchmarking (default: H200)")
    parser.add_argument("--baseline", default="baseline_time_torch",
                        help="Baseline to compare against (default: baseline_time_torch)")

    # Pipeline Control
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip the generation step")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip the evaluation step")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip the analysis step")
    parser.add_argument("--skip-server-check", action="store_true",
                        help="Skip server health check")

    args = parser.parse_args()

    # Generate default run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"qwen3_235b_run_{timestamp}"

    return args


# Get the KernelBench directory
SCRIPT_DIR = Path(__file__).parent.resolve()
KERNELBENCH_DIR = SCRIPT_DIR.parent


def main():
    """Main pipeline execution"""
    args = parse_args()

    print_header("KernelBench Pipeline - Qwen3-235B-A22B")

    print("Configuration:")
    print(f"  Run Name:        {args.run_name}")
    print(f"  Dataset Source:  {args.dataset_src}")
    print(f"  Level:           {args.level}")
    print(f"  Server:          http://{args.server_address}:{args.server_port}")
    print(f"  Model:           {args.model_name}")
    print(f"  Num Workers:     {args.num_workers}")
    print(f"  GPU Devices:     {args.num_gpu_devices}")
    print(f"  Hardware:        {args.hardware}")
    if args.subset:
        print(f"  Subset:          {args.subset[0]}-{args.subset[1]}")
    print()

    # Check server health
    if not args.skip_server_check:
        if not check_server(args.server_address, args.server_port):
            sys.exit(1)

    # Step 1: Generate samples
    if not args.skip_generation:
        if not generate_samples(args):
            sys.exit(1)
    else:
        print_warning("Skipping generation step")

    # Step 2: Evaluate samples
    if not args.skip_evaluation:
        if not evaluate_samples(args):
            sys.exit(1)
    else:
        print_warning("Skipping evaluation step")

    # Step 3: Analyze results
    if not args.skip_analysis:
        if not analyze_results(args):
            sys.exit(1)
    else:
        print_warning("Skipping analysis step")

    # Print summary
    print_summary(args)


if __name__ == "__main__":
    main()
