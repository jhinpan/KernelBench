# GLM-5.2-FP8 on KernelBench (Triton) — AMD MI300X

**Date:** 2026-06-22 · **Branch:** `MI300-GLM5.2` · **Hardware:** 8× AMD Instinct MI300X (gfx942), ROCm 7.2

## Task
Use **GLM-5.2-FP8** (served by SGLang, OpenAI endpoint on `:30000`) to generate **Triton** GPU kernels for KernelBench PyTorch programs, and measure how many the model can "serve" (compile + pass correctness) on MI300X — across all of Levels 1–3.

## Setup
- **Generator:** GLM-5.2-FP8 via SGLang, wired as KernelBench `server_type=glm_5_2` (routed through `/v1/chat/completions` so the chat template + thinking apply; inline `<think>…</think>` is stripped before code extraction). Greedy (temp 0), `max_tokens=16384`, one-shot prompt.
- **Backend:** `backend=triton`, `precision=fp32`, `gpu_arch=["gfx942"]`. Triton runs on ROCm via its HIP backend; KernelBench eval explicitly allows `triton` on AMD.
- **Eval:** serial mode (`glm_5_2_eval_serial.sh`) for ROCm stability — per-problem, `num_correct_trials=5`, `num_perf_trials=30`, 180 s timeout, skip-record on hang. Server shut down during eval to give the GPU full memory.
- **"Served" = compiled AND correct** (passes all 5 randomized-input trials vs the PyTorch reference, `torch.allclose` fp32 tol 1e-4). **fast_1** = correct AND faster than PyTorch eager.

## Results — all 3 levels (250 problems)

| Level | Problems | Generated valid Triton | Compiled | **Correct (served)** | fast_1 | Skipped (eval hang) |
|-------|---------:|-----------------------:|---------:|---------------------:|-------:|--------------------:|
| **L1** single-op | 100 | 89 | 41 | **29** (29%) | 0 | 3 |
| **L2** fusion | 100 | 80 | 33 | **20** (20%) | 0 | 4 |
| **L3** full model | 50 | 36 | 2 | **1** (2%) | 0 | 5 |
| **Total** | **250** | **205** (82%) | **76** | **50 (20%)** | **0** | 12 |

Funnel (all levels): **250 tasks → 205 generated a valid `@triton.jit` kernel → 76 compiled on MI300X → 50 numerically correct → 0 faster than PyTorch.**

## Reading the numbers
- **Generation is the model's strength:** GLM-5.2 produced a syntactically-valid Triton kernel (`@triton.jit` + `tl.*`, passes the static checker) for **82%** of tasks. The 45 generation failures were ops where it returned PyTorch / a `pass` / no `@triton.jit` — concentrated in reductions (cumsum) and loss functions.
- **Correctness drops with difficulty, as expected:** L1 29% → L2 20% → L3 **2%**. Full model architectures (L3) in hand-written Triton are extremely hard — only 1/50 correct, and only 2/36 generated kernels even compiled (most L3 attempts reference undefined symbols / partial kernels).
- **Dominant failure modes:** compile-stage `NameError`/`AttributeError` (undefined names, missing `ModelNew` for the static-rejected problems), then `Output mismatch` on the kernels that did compile.
- **fast_1 = 0 everywhere:** no one-shot generated Triton kernel beat PyTorch eager (which dispatches to tuned rocBLAS/aten on MI300X). This matches public KernelBench findings — speedups need iterative/agentic refinement, not single-shot generation.

## How to reproduce
```bash
# 1. Serve GLM-5.2-FP8 (SGLang, ROCm) on :30000  (see sglang-cookbook glm52_fp8_playbook)
# 2. KernelBench generate (per level L in 1 2 3):
SGLANG_API_KEY=EMPTY LEVEL=$L NUM_WORKERS=16 bash scripts/glm_5_2_generate.sh
# 3. Eval on MI300X (server can be down):
LEVEL=$L bash scripts/glm_5_2_eval_serial.sh
# 4. Summary:
python scripts/glm_5_2_report.py --run-names glm_5_2_level1,glm_5_2_level2,glm_5_2_level3
```

## Files added on this branch
- `src/kernelbench/utils.py` — `glm_5_2` server type + preset; chat routing + `<think>` stripping.
- `scripts/glm_5_2_{smoke,generate,eval,eval_serial,all_levels}.sh`, `scripts/glm_5_2_report.py`.

## Caveats
- One-shot, greedy, single sample per problem (no pass@k, no self-repair). Numbers are a floor for what GLM-5.2 can do with more samples/iteration.
- `max_tokens=16384` with GLM-5.2's default `max` reasoning effort occasionally truncates before the code block on the hardest problems (contributes to L3 generation misses).
- Triton-on-ROCm numeric tolerance (fp32 1e-4) can flag MI300X matmul drift as "Output mismatch".
