#!/usr/bin/env python3
"""
generate_kb_prompts_jsonl.py

Create a JSONL file with (instruction, input, output) triples for every
problem in KernelBench level-1 on Hugging Face.  The "input" field is the full
prompt produced by `prompt_generate_custom_cuda_from_prompt_template`, the
"instruction" is a short system message, and "output" is left empty because
we are *not* generating answers yet.

Usage (from repo root):

  python3 scripts/generate_kb_prompts_jsonl.py \
      --output kb_hf_level_1.jsonl \
      --dataset-name ScalingIntelligence/KernelBench \
      --level 1
"""
import argparse, json, os, sys
from datasets import load_dataset

# repo-local import (works the same way as in generate_samples.py)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

def build_jsonl(dataset_name: str, level: int, output_path: str) -> None:
    ds = load_dataset(dataset_name)
    level_key = f"level_{level}"
    if level_key not in ds:
        raise ValueError(f"{dataset_name} has no split named {level_key}")

    level_ds = ds[level_key]
    system_prompt = (
        "You are an expert in writing CUDA Kernels for efficient GPU programming."
    )

    with open(output_path, "w") as f:
        for row in level_ds:
            problem_id = row["problem_id"]       # already 1-based
            ref_arch_src = row["code"]           # PyTorch reference implementation

            full_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

            record = {
                "instruction": system_prompt,
                "input": full_prompt,
                "output": "",                    # left blank intentionally
                "problem_id": problem_id         # keeps provenance handy
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(level_ds)} prompts to {output_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--output", default="kb_hf_level_1.jsonl")
    args = ap.parse_args()

    build_jsonl(args.dataset_name, args.level, args.output)

if __name__ == "__main__":
    main()