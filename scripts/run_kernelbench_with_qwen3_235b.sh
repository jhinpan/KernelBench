#!/bin/bash

################################################################################
# KernelBench Full Pipeline with Qwen3-235B-A22B (localhost:30000)
################################################################################
# This script runs the complete KernelBench pipeline:
# 1. Generate kernel implementations using the model at localhost:30000
# 2. Evaluate the generated kernels (correctness + performance)
# 3. Analyze results and compute metrics
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Model Server Configuration
SERVER_TYPE="local"
SERVER_ADDRESS="localhost"
SERVER_PORT="30000"
MODEL_NAME="Qwen/Qwen3-235B-A22B"

# Run Configuration
RUN_NAME="${RUN_NAME:-qwen3_235b_run_$(date +%Y%m%d_%H%M%S)}"
DATASET_SRC="${DATASET_SRC:-local}"
LEVEL="${LEVEL:-1}"  # Difficulty level: 1-4
NUM_WORKERS="${NUM_WORKERS:-4}"  # Parallel workers for generation
NUM_GPU_DEVICES="${NUM_GPU_DEVICES:-2}"  # GPUs for evaluation (set to 2 since we have 2 available)
HARDWARE="${HARDWARE:-H200}"  # Hardware type for benchmarking
BASELINE="${BASELINE:-baseline_time_torch}"  # Baseline to compare against

# Advanced Options
NUM_SAMPLES="${NUM_SAMPLES:-1}"  # Samples per problem
NUM_TRIALS_CORRECTNESS="${NUM_TRIALS_CORRECTNESS:-5}"  # Correctness trials
NUM_TRIALS_PERFORMANCE="${NUM_TRIALS_PERFORMANCE:-100}"  # Performance trials

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

check_server() {
    print_info "Checking if model server is running at ${SERVER_ADDRESS}:${SERVER_PORT}..."

    if curl -s "http://${SERVER_ADDRESS}:${SERVER_PORT}/health" > /dev/null 2>&1; then
        print_info "✓ Server is responding"
        return 0
    elif curl -s "http://${SERVER_ADDRESS}:${SERVER_PORT}/v1/models" > /dev/null 2>&1; then
        print_info "✓ Server is responding (OpenAI API)"
        return 0
    else
        print_error "✗ Server is not responding at http://${SERVER_ADDRESS}:${SERVER_PORT}"
        print_error "Please ensure your model server is running before proceeding."
        print_error "You can start it with: ./launch_qwen3_235b_a22b_server.sh"
        return 1
    fi
}

# ============================================================================
# Main Pipeline
# ============================================================================

print_header "KernelBench Pipeline - Qwen3-235B-A22B"

echo "Configuration:"
echo "  Run Name:        $RUN_NAME"
echo "  Dataset Source:  $DATASET_SRC"
echo "  Level:           $LEVEL"
echo "  Server:          http://${SERVER_ADDRESS}:${SERVER_PORT}"
echo "  Model:           $MODEL_NAME"
echo "  Num Workers:     $NUM_WORKERS"
echo "  GPU Devices:     $NUM_GPU_DEVICES"
echo "  Hardware:        $HARDWARE"
echo ""

# Check if server is running
if ! check_server; then
    exit 1
fi

# Navigate to KernelBench directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELBENCH_DIR="$(dirname "$SCRIPT_DIR")"
cd "$KERNELBENCH_DIR"
print_info "Working directory: $(pwd)"

# ============================================================================
# Step 1: Generate Samples
# ============================================================================

print_header "Step 1/3: Generating Kernel Samples"

print_info "Starting generation with ${NUM_WORKERS} workers..."
print_info "This may take a while depending on the number of problems and model speed..."

python3 scripts/generate_samples.py \
    run_name="$RUN_NAME" \
    dataset_src="$DATASET_SRC" \
    level="$LEVEL" \
    server_type="$SERVER_TYPE" \
    server_address="$SERVER_ADDRESS" \
    server_port="$SERVER_PORT" \
    model_name="$MODEL_NAME" \
    num_workers="$NUM_WORKERS" \
    num_samples="$NUM_SAMPLES"

if [ $? -ne 0 ]; then
    print_error "Generation failed!"
    exit 1
fi

print_info "✓ Generation completed successfully"
print_info "Generated samples stored in: runs/${RUN_NAME}/"

# ============================================================================
# Step 2: Evaluate Generated Kernels
# ============================================================================

print_header "Step 2/3: Evaluating Generated Kernels"

print_info "Starting evaluation with ${NUM_GPU_DEVICES} GPU devices..."
print_info "Correctness trials: ${NUM_TRIALS_CORRECTNESS}"
print_info "Performance trials: ${NUM_TRIALS_PERFORMANCE}"

python3 scripts/eval_from_generations.py \
    run_name="$RUN_NAME" \
    dataset_src="$DATASET_SRC" \
    level="$LEVEL" \
    num_gpu_devices="$NUM_GPU_DEVICES" \
    n_correct="$NUM_TRIALS_CORRECTNESS" \
    n_trials="$NUM_TRIALS_PERFORMANCE"

if [ $? -ne 0 ]; then
    print_error "Evaluation failed!"
    exit 1
fi

print_info "✓ Evaluation completed successfully"
print_info "Evaluation results stored in: runs/${RUN_NAME}/eval_results.json"

# ============================================================================
# Step 3: Analyze Results
# ============================================================================

print_header "Step 3/3: Analyzing Benchmark Results"

print_info "Computing metrics and performance analysis..."

python3 scripts/benchmark_eval_analysis.py \
    run_name="$RUN_NAME" \
    level="$LEVEL" \
    hardware="$HARDWARE" \
    baseline="$BASELINE"

if [ $? -ne 0 ]; then
    print_error "Analysis failed!"
    exit 1
fi

print_info "✓ Analysis completed successfully"

# ============================================================================
# Summary
# ============================================================================

print_header "Pipeline Completed Successfully!"

echo "Results Summary:"
echo "  Run Name:     $RUN_NAME"
echo "  Location:     runs/${RUN_NAME}/"
echo "  Generations:  runs/${RUN_NAME}/generations/"
echo "  Evaluations:  runs/${RUN_NAME}/eval_results.json"
echo ""
echo "To view detailed results:"
echo "  cat runs/${RUN_NAME}/eval_results.json"
echo ""
echo "To inspect specific problem generations:"
echo "  ls runs/${RUN_NAME}/generations/"
echo ""

print_info "All done! 🎉"
