import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, read_file, create_inference_server_from_presets

"""
Generate a single sample (generation only, no evaluation)
Designed to work with external SGLang server (e.g., running in Docker)

Launching SGLang server in Docker:
```
docker run -p 30000:30000 your-sglang-image python -m src.server.sglang_server --model_name deepseek-ai/deepseek-coder-6.7b-instruct --port 30000
```

Using this script to generate:
```
python3 scripts/generate_single_sample.py dataset_src="huggingface" level=2 problem_id=40 model_name="deepseek-ai/deepseek-coder-6.7b-instruct"
```

The generated kernel will be saved to runs/{run_name}/level_{level}/problem_{problem_id}/kernel.py
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class GenerationConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Run name for organizing outputs
        self.run_name = REQUIRED

        # Inference config
        self.server_type = "sglang"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0

        # Output directory
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        
        # Logging
        self.verbose = False
        self.log_prompt = False

    def __repr__(self):
        return f"GenerationConfig({self.to_dict()})"


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Generate a single kernel sample without evaluation
    """
    print(f"Starting Generation with config: {config}")

    # Load dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Starting Generation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Problem
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
    
    # 2. Setup output directory
    output_dir = os.path.join(config.runs_dir, config.run_name, f"level_{config.level}", f"problem_{config.problem_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Save reference architecture for record keeping
    with open(os.path.join(output_dir, "reference.py"), "w") as f:
        f.write(ref_arch_src)

    # Save problem metadata
    metadata = {
        "problem_id": config.problem_id,
        "problem_name": problem_name,
        "level": config.level,
        "dataset_src": config.dataset_src,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 3. Generate Sample
    # Create inference function with config parameters
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )

    # Generate prompt
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    
    if config.log_prompt:
        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    print("Querying inference server...")
    custom_cuda_raw = inference_server(custom_cuda_prompt)
    
    # Save raw output
    with open(os.path.join(output_dir, "raw_output.txt"), "w") as f:
        f.write(custom_cuda_raw)
    
    print("=====================================")
    print("Raw output from model:")
    print(custom_cuda_raw)
    print("=====================================")
    
    # Extract code
    custom_cuda = extract_first_code(custom_cuda_raw, ["python", "cpp"])
    
    # Check LLM is able to generate custom CUDA code
    if custom_cuda is None:
        print("ERROR: Failed to extract CUDA code from model output")
        # Save empty kernel file to indicate failure
        with open(os.path.join(output_dir, "kernel.py"), "w") as f:
            f.write("# GENERATION FAILED: Could not extract CUDA code from model output\n")
        return
    
    # Save generated kernel
    with open(os.path.join(output_dir, "kernel.py"), "w") as f:
        f.write(custom_cuda)
    
    print(f"✓ Generated kernel saved to: {os.path.join(output_dir, 'kernel.py')}")


if __name__ == "__main__":
    main()