# GLM-5.2-FP8 on KernelBench (Triton) — AMD MI300X

**Date:** 2026-06-23 · **Branch:** `MI300-GLM5.2` · **Hardware:** 8× AMD Instinct MI300X (gfx942), ROCm 7.2

## Task
Use **GLM-5.2-FP8** (served by SGLang, OpenAI endpoint on `:30000`) to generate **Triton** GPU kernels for KernelBench PyTorch programs, and measure how many the model can "serve" (**compile + pass correctness**) on MI300X — across Levels 1–3 (250 problems).

## Setup
- **Generator:** GLM-5.2-FP8 via SGLang, `server_type=glm_5_2`, routed through `/v1/chat/completions` (chat template + thinking applied; inline `<think>…</think>` stripped). Greedy (temp 0), `max_tokens=16384`, one-shot prompt, **one sample/problem**.
- **Backend:** `backend=triton`, `precision=fp32`, `gpu_arch=["gfx942"]`. Triton runs on ROCm via its HIP backend.
- **Eval:** serial mode (ROCm-stable), `num_correct_trials=5`, `num_perf_trials=30`, 180 s timeout, server down during eval.
- **"Served" = compiled AND correct** (5/5 randomized trials vs PyTorch, fp32 tol 1e-4). **fast_1** = correct AND faster than PyTorch eager.

## Headline results (corrected)

| Level | Problems | Generated valid Triton | Compiled | **Correct (served)** | fast_1 |
|-------|---------:|-----------------------:|---------:|---------------------:|-------:|
| **L1** single-op | 100 | 88 | 84 | **51 (51%)** | 0 |
| **L2** fusion | 100 | 87 | 78 | **35 (35%)** | 0 |
| **L3** full model | 50 | 16 | 16 | **6 (12%)** | 0 |
| **Total** | **250** | **191** | **178 (71%)** | **92 (37%)** | **0** |

## Important: a harness bug was deflating round-1 results
The first run scored only **50/250 (20%)**. Root cause was **not** GLM-5.2 — it was KernelBench's `extract_first_code`, which grabs the **first** ```` ```python ```` block. GLM-5.2 (a thinking model) writes an illustrative kernel *snippet* first and the **complete module** (imports + `@triton.jit` + `class ModelNew`) later, so ~half the saved kernels were snippets missing imports/`ModelNew` → instant `NameError`.

**Fix:** `extract_code_for_modelnew()` selects the block containing `ModelNew` (+ raw responses are now saved). Because generation is temp 0 (deterministic), re-running recovered the true kernels.

### Before vs after (same model, same server, same settings)
| | Compiled | Correct | Correct % |
|---|---:|---:|---:|
| Round 1 (`extract_first_code` bug) | 76/250 | 50/250 | 20% |
| **Corrected (`extract_code_for_modelnew`)** | **178/250** | **92/250** | **37%** |

Per level, correct went L1 29→**51**, L2 20→**35**, L3 1→**6**. (L3 "generated" dropped 36→16: the corrected extraction returns full modules, some of which KernelBench's static checker rejects for using `torch.nn` layers; round-1's 36 were mostly junk snippets that happened to pass the checker.)

## Reading the numbers
- **Generation is GLM-5.2's strength:** valid Triton kernels for L1/L2 ~87–88%. L3 (full architectures) is much harder to express purely in Triton.
- **Correctness drops with difficulty:** L1 51% → L2 35% → L3 12%. This is in line with public KernelBench (single-shot, no iteration).
- **fast_1 = 0 across all levels:** no one-shot generated Triton kernel beat PyTorch eager (tuned rocBLAS/aten on MI300X). Speedups need iterative/agentic refinement, not single greedy samples.

## Reproduce
```bash
# serve GLM-5.2-FP8 (SGLang, ROCm) on :30000, then per level L in 1 2 3:
SGLANG_API_KEY=EMPTY LEVEL=$L NUM_WORKERS=16 bash scripts/glm_5_2_generate.sh
LEVEL=$L bash scripts/glm_5_2_eval_serial.sh      # server can be down
python scripts/glm_5_2_report.py --run-names glm_5_2_level1,glm_5_2_level2,glm_5_2_level3
```

## Files on this branch
- `src/kernelbench/utils.py` — `glm_5_2` server type + preset; chat routing + `<think>` strip; `extract_code_for_modelnew`.
- `scripts/glm_5_2_*.sh`, `scripts/glm_5_2_report.py`.
- `results/glm5.2_mi300x_corrected/` (corrected per-level verdicts + summary); `results/glm5.2_mi300x/` (round-1, for the record).

## Caveats
One-shot, greedy, single sample/problem (no pass@k, no self-repair) — a floor, not a ceiling. `max_tokens=16384` with GLM-5.2's default `max` reasoning occasionally truncates the hardest problems before the code block.
