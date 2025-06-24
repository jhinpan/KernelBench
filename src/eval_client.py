"""
Client library for interacting with KernelBench evaluation service.

This can be used by the RL framework to evaluate generated kernels.
"""

import requests
import json
from typing import List, Dict, Optional, Union


class KernelBenchEvalClient:
    """Client for KernelBench evaluation service."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the KernelBench evaluation service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the service is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def evaluate_kernel(
        self,
        dataset_src: str,
        level: int,
        problem_id: int,
        kernel_code: str,
        dataset_name: str = "ScalingIntelligence/KernelBench",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        gpu_arch: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a single kernel.
        
        Args:
            dataset_src: "huggingface" or "local"
            level: Problem level (1-4)
            problem_id: Problem ID within the level
            kernel_code: Generated CUDA kernel code
            dataset_name: Name of the dataset (for huggingface)
            num_correct_trials: Number of correctness trials
            num_perf_trials: Number of performance trials
            gpu_arch: GPU architecture list (e.g., ["Hopper"])
            verbose: Whether to enable verbose logging
            
        Returns:
            Evaluation results dictionary
        """
        payload = {
            "dataset_src": dataset_src,
            "dataset_name": dataset_name,
            "level": level,
            "problem_id": problem_id,
            "kernel_code": kernel_code,
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "verbose": verbose
        }
        
        if gpu_arch is not None:
            payload["gpu_arch"] = gpu_arch
        
        response = self.session.post(
            f"{self.base_url}/evaluate",
            json=payload,
            timeout=300  # 5 minute timeout for evaluation
        )
        response.raise_for_status()
        return response.json()
    
    def batch_evaluate(
        self,
        evaluations: List[Dict],
        common_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Evaluate multiple kernels in a single request.
        
        Args:
            evaluations: List of evaluation requests
            common_params: Common parameters to apply to all evaluations
            
        Returns:
            List of evaluation results
        """
        payload = {
            "evaluations": evaluations
        }
        
        if common_params:
            payload["common_params"] = common_params
        
        response = self.session.post(
            f"{self.base_url}/batch_evaluate",
            json=payload,
            timeout=1800  # 30 minute timeout for batch evaluation
        )
        response.raise_for_status()
        return response.json()["results"]


# Helper functions for common use cases

def evaluate_single_kernel(
    kernelbench_url: str,
    level: int,
    problem_id: int,
    kernel_code: str,
    **kwargs
) -> Dict:
    """
    Convenience function to evaluate a single kernel.
    
    Args:
        kernelbench_url: URL of the KernelBench service
        level: Problem level
        problem_id: Problem ID
        kernel_code: Generated kernel code
        **kwargs: Additional parameters for evaluation
        
    Returns:
        Evaluation result
    """
    client = KernelBenchEvalClient(kernelbench_url)
    return client.evaluate_kernel(
        dataset_src=kwargs.get("dataset_src", "huggingface"),
        level=level,
        problem_id=problem_id,
        kernel_code=kernel_code,
        **kwargs
    )


def evaluate_rollout_batch(
    kernelbench_url: str,
    rollout_results: List[Dict[str, Union[int, str]]],
    common_params: Optional[Dict] = None
) -> List[Dict]:
    """
    Evaluate a batch of rollout results.
    
    Args:
        kernelbench_url: URL of the KernelBench service
        rollout_results: List of dicts with keys: level, problem_id, kernel_code
        common_params: Common evaluation parameters
        
    Returns:
        List of evaluation results
    """
    client = KernelBenchEvalClient(kernelbench_url)
    
    # Convert rollout results to evaluation requests
    evaluations = []
    for result in rollout_results:
        eval_request = {
            "dataset_src": "huggingface",
            "level": result["level"],
            "problem_id": result["problem_id"],
            "kernel_code": result["kernel_code"]
        }
        evaluations.append(eval_request)
    
    return client.batch_evaluate(evaluations, common_params)


# Example usage for RL framework
if __name__ == "__main__":
    # Example: Evaluate a single kernel
    client = KernelBenchEvalClient("http://kernelbench-service:8080")
    
    # Check service health
    health = client.health_check()
    print(f"Service health: {health}")
    
    # Example kernel code (would come from SGLang rollout)
    example_kernel = '''
import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * x
    '''
    
    # Evaluate the kernel
    result = client.evaluate_kernel(
        dataset_src="huggingface",
        level=1,
        problem_id=1,
        kernel_code=example_kernel,
        num_correct_trials=5,
        num_perf_trials=100
    )
    
    print(f"Evaluation result: {json.dumps(result, indent=2)}")