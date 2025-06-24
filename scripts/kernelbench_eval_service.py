"""
KernelBench Evaluation Service

This service runs inside the KernelBench Docker container and provides
an API endpoint for evaluating generated CUDA kernels.

Usage:
    python scripts/kernelbench_eval_service.py --port 8080
"""

import argparse
import json
import logging
import traceback
from flask import Flask, request, jsonify
import torch
from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.utils import set_gpu_arch, read_file

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for dataset to avoid reloading
DATASET_CACHE = {}

def get_reference_code(dataset_src, dataset_name, level, problem_id):
    """Get reference PyTorch code for a given problem."""
    cache_key = f"{dataset_src}_{dataset_name}_{level}"
    
    if cache_key not in DATASET_CACHE:
        if dataset_src == "huggingface":
            dataset = load_dataset(dataset_name)
            DATASET_CACHE[cache_key] = dataset[f"level_{level}"]
        elif dataset_src == "local":
            DATASET_CACHE[cache_key] = construct_kernelbench_dataset(level)
    
    curr_level_dataset = DATASET_CACHE[cache_key]
    
    if dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == problem_id)
        if len(curr_problem_row) == 0:
            raise ValueError(f"Problem ID {problem_id} not found in level {level}")
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif dataset_src == "local":
        problem_idx = problem_id - 1  # 0-indexed locally
        if problem_idx >= len(curr_level_dataset):
            raise ValueError(f"Problem ID {problem_id} out of range for level {level}")
        ref_arch_path = curr_level_dataset[problem_idx]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    
    return ref_arch_src, problem_name


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "kernelbench_eval",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate a CUDA kernel against reference PyTorch code.
    
    Expected JSON payload:
    {
        "dataset_src": "huggingface" or "local",
        "dataset_name": "ScalingIntelligence/KernelBench",  # for huggingface
        "level": 1-4,
        "problem_id": int,
        "kernel_code": str,  # The generated CUDA kernel code
        "num_correct_trials": 5,  # optional
        "num_perf_trials": 100,  # optional
        "gpu_arch": ["Hopper"],  # optional
        "verbose": false  # optional
    }
    
    Returns:
    {
        "success": bool,
        "problem_name": str,
        "correctness": bool,
        "runtime_ms": float or null,
        "ref_runtime_ms": float or null,
        "speedup": float or null,
        "error": str or null
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ["dataset_src", "level", "problem_id", "kernel_code"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract parameters
        dataset_src = data["dataset_src"]
        dataset_name = data.get("dataset_name", "ScalingIntelligence/KernelBench")
        level = data["level"]
        problem_id = data["problem_id"]
        kernel_code = data["kernel_code"]
        num_correct_trials = data.get("num_correct_trials", 5)
        num_perf_trials = data.get("num_perf_trials", 100)
        gpu_arch = data.get("gpu_arch", ["Hopper"])
        verbose = data.get("verbose", False)
        
        logger.info(f"Evaluating kernel for level {level}, problem {problem_id}")
        
        # Set GPU architecture
        if gpu_arch:
            set_gpu_arch(gpu_arch)
        
        # Get reference code
        ref_arch_src, problem_name = get_reference_code(
            dataset_src, dataset_name, level, problem_id
        )
        
        # Evaluate kernel
        result = eval_kernel_against_ref(
            ref_arch_src,
            kernel_code,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials
        )
        
        # Prepare response
        response = {
            "success": result.success,
            "problem_name": problem_name,
            "correctness": result.correctness,
            "runtime_ms": result.runtime_ms,
            "ref_runtime_ms": result.ref_runtime_ms,
            "speedup": result.speedup,
            "error": result.error
        }
        
        logger.info(f"Evaluation complete: success={result.success}, speedup={result.speedup}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """
    Evaluate multiple kernels in a single request.
    
    Expected JSON payload:
    {
        "evaluations": [
            {
                "dataset_src": "huggingface",
                "level": 1,
                "problem_id": 1,
                "kernel_code": "...",
                ...
            },
            ...
        ],
        "common_params": {  # optional, applied to all evaluations
            "num_correct_trials": 5,
            "num_perf_trials": 100,
            "gpu_arch": ["Hopper"]
        }
    }
    
    Returns:
    {
        "results": [
            {evaluation_result_1},
            {evaluation_result_2},
            ...
        ]
    }
    """
    try:
        data = request.json
        evaluations = data.get("evaluations", [])
        common_params = data.get("common_params", {})
        
        results = []
        
        for eval_request in evaluations:
            # Merge common params with individual params
            merged_request = {**common_params, **eval_request}
            
            # Call single evaluation endpoint logic
            try:
                # Get reference code
                ref_arch_src, problem_name = get_reference_code(
                    merged_request["dataset_src"],
                    merged_request.get("dataset_name", "ScalingIntelligence/KernelBench"),
                    merged_request["level"],
                    merged_request["problem_id"]
                )
                
                # Evaluate
                result = eval_kernel_against_ref(
                    ref_arch_src,
                    merged_request["kernel_code"],
                    verbose=merged_request.get("verbose", False),
                    measure_performance=True,
                    num_correct_trials=merged_request.get("num_correct_trials", 5),
                    num_perf_trials=merged_request.get("num_perf_trials", 100)
                )
                
                results.append({
                    "level": merged_request["level"],
                    "problem_id": merged_request["problem_id"],
                    "success": result.success,
                    "problem_name": problem_name,
                    "correctness": result.correctness,
                    "runtime_ms": result.runtime_ms,
                    "ref_runtime_ms": result.ref_runtime_ms,
                    "speedup": result.speedup,
                    "error": result.error
                })
                
            except Exception as e:
                results.append({
                    "level": merged_request.get("level"),
                    "problem_id": merged_request.get("problem_id"),
                    "error": str(e),
                    "success": False
                })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}")
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def main():
    parser = argparse.ArgumentParser(description="KernelBench Evaluation Service")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the service on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting KernelBench Evaluation Service on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    import os
    import sys
    # Add parent directory to path to import src modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    main()