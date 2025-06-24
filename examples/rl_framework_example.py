"""
Example of how to integrate KernelBench evaluation with an RL framework.

This shows how the RL framework Docker would:
1. Generate kernels using SGLang servers (rollout)
2. Evaluate them using KernelBench Docker
3. Use the results as rewards for RL training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_client import KernelBenchEvalClient, evaluate_rollout_batch
import json
import time
from typing import List, Dict
import requests


class SGLangRolloutClient:
    """Client for interacting with SGLang servers for rollout."""
    
    def __init__(self, sglang_urls: List[str]):
        self.sglang_urls = sglang_urls
        self.current_server = 0
    
    def generate_kernel(self, prompt: str, server_url: str = None) -> str:
        """Generate a kernel using SGLang server."""
        if server_url is None:
            # Round-robin server selection
            server_url = self.sglang_urls[self.current_server]
            self.current_server = (self.current_server + 1) % len(self.sglang_urls)
        
        # This is a simplified example - adjust based on your SGLang API
        response = requests.post(
            f"{server_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["text"]


class RLKernelOptimizer:
    """Example RL framework for kernel optimization."""
    
    def __init__(
        self,
        kernelbench_url: str = "http://localhost:8080",
        sglang_urls: List[str] = None
    ):
        self.eval_client = KernelBenchEvalClient(kernelbench_url)
        self.rollout_client = SGLangRolloutClient(sglang_urls or ["http://localhost:30000"])
        
        # Example: Simple problem selection strategy
        self.problem_pool = [
            {"level": 1, "problem_id": i} for i in range(1, 11)
        ] + [
            {"level": 2, "problem_id": i} for i in range(1, 6)
        ]
    
    def generate_prompt_for_problem(self, level: int, problem_id: int) -> str:
        """Generate prompt for a specific problem."""
        # In practice, you would fetch the actual PyTorch reference code
        # This is just an example
        return f"""
Convert the following PyTorch model to an optimized CUDA kernel:

```python
# Level {level}, Problem {problem_id}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * x  # Example operation
```

Generate only the CUDA kernel code.
"""
    
    def run_episode(self, num_rollouts: int = 10) -> Dict:
        """Run one episode of RL training."""
        print(f"\n=== Running episode with {num_rollouts} rollouts ===")
        
        # 1. Select problems for this episode
        import random
        selected_problems = random.sample(self.problem_pool, min(num_rollouts, len(self.problem_pool)))
        
        # 2. Generate rollouts (kernels) using SGLang
        print("Generating kernels...")
        rollouts = []
        for problem in selected_problems:
            prompt = self.generate_prompt_for_problem(problem["level"], problem["problem_id"])
            
            try:
                # Generate kernel using SGLang
                generated_code = self.rollout_client.generate_kernel(prompt)
                
                rollouts.append({
                    "level": problem["level"],
                    "problem_id": problem["problem_id"],
                    "kernel_code": generated_code,
                    "prompt": prompt
                })
                print(f"  ✓ Generated kernel for L{problem['level']} P{problem['problem_id']}")
            except Exception as e:
                print(f"  ✗ Failed to generate for L{problem['level']} P{problem['problem_id']}: {e}")
        
        # 3. Batch evaluate all generated kernels
        print("\nEvaluating kernels...")
        eval_results = evaluate_rollout_batch(
            self.eval_client.base_url,
            rollouts,
            common_params={
                "num_correct_trials": 5,
                "num_perf_trials": 50,  # Reduced for faster demo
                "gpu_arch": ["Hopper"]
            }
        )
        
        # 4. Calculate rewards and statistics
        rewards = []
        successful_kernels = 0
        total_speedup = 0.0
        
        for i, result in enumerate(eval_results):
            problem = selected_problems[i]
            
            if result.get("success") and result.get("correctness"):
                # Reward based on speedup
                speedup = result.get("speedup", 1.0)
                reward = max(0, speedup - 1.0)  # Reward for speedup > 1x
                successful_kernels += 1
                total_speedup += speedup
                print(f"  ✓ L{problem['level']} P{problem['problem_id']}: "
                      f"Speedup={speedup:.2f}x, Reward={reward:.3f}")
            else:
                # Penalty for incorrect or failed kernels
                reward = -1.0
                error = result.get("error", "Unknown error")
                print(f"  ✗ L{problem['level']} P{problem['problem_id']}: Failed - {error}")
            
            rewards.append(reward)
        
        # 5. Episode statistics
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success_rate = successful_kernels / len(rollouts) if rollouts else 0
        avg_speedup = total_speedup / successful_kernels if successful_kernels > 0 else 0
        
        episode_stats = {
            "num_rollouts": len(rollouts),
            "successful_kernels": successful_kernels,
            "success_rate": success_rate,
            "average_reward": avg_reward,
            "average_speedup": avg_speedup,
            "rewards": rewards
        }
        
        print(f"\nEpisode Summary:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        
        # 6. In a real RL framework, you would update your policy here
        # self.update_policy(rollouts, rewards)
        
        return episode_stats
    
    def train(self, num_episodes: int = 5):
        """Run multiple episodes of training."""
        print("Starting RL training for kernel optimization")
        
        # Check service health
        try:
            health = self.eval_client.health_check()
            print(f"KernelBench service status: {health['status']}")
            print(f"GPUs available: {health['gpu_count']}")
        except Exception as e:
            print(f"Warning: Could not connect to KernelBench service: {e}")
            print("Make sure the service is running!")
            return
        
        # Training loop
        all_stats = []
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            stats = self.run_episode(num_rollouts=5)
            all_stats.append(stats)
            
            # Save intermediate results
            with open(f"episode_{episode + 1}_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            time.sleep(1)  # Brief pause between episodes
        
        # Final summary
        print(f"\n{'='*50}")
        print("Training Complete!")
        print(f"{'='*50}")
        
        total_success_rate = sum(s["success_rate"] for s in all_stats) / len(all_stats)
        total_avg_speedup = sum(s["average_speedup"] for s in all_stats) / len(all_stats)
        
        print(f"Overall Success Rate: {total_success_rate:.1%}")
        print(f"Overall Average Speedup: {total_avg_speedup:.2f}x")


def main():
    """Example usage."""
    
    # Configure based on your Docker setup
    optimizer = RLKernelOptimizer(
        kernelbench_url="http://localhost:8080",  # KernelBench eval service
        sglang_urls=[
            "http://localhost:30001",  # SGLang server 1
            "http://localhost:30002",  # SGLang server 2
        ]
    )
    
    # Note: This is a simplified example that won't actually work without:
    # 1. Real SGLang servers running
    # 2. KernelBench evaluation service running
    # 3. Proper prompt generation with actual PyTorch reference code
    
    print("This is a demonstration of how to structure the RL framework.")
    print("To actually run this, you need:")
    print("1. KernelBench evaluation service running on port 8080")
    print("2. SGLang servers running on ports 30001, 30002")
    print("3. Proper integration with your RL framework")
    
    # Uncomment to run when services are available:
    # optimizer.train(num_episodes=3)


if __name__ == "__main__":
    main()