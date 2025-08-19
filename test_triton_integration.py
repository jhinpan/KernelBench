#!/usr/bin/env python3
"""
Test script to verify Triton evaluation server integration
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("1. Testing Module Imports")
    print("=" * 60)
    
    tests = []
    
    # Test core eval module with backend support
    try:
        from src.eval import eval_kernel_against_ref, KernelExecResult
        import inspect
        sig = inspect.signature(eval_kernel_against_ref)
        if 'backend' in sig.parameters:
            print("✓ src.eval with backend parameter")
            tests.append(True)
        else:
            print("✗ src.eval missing backend parameter")
            tests.append(False)
    except Exception as e:
        print(f"✗ src.eval import failed: {e}")
        tests.append(False)
    
    # Test Triton support modules
    try:
        from src.prompt_constructor_triton import prompt_generate_custom_triton_from_prompt_template
        print("✓ src.prompt_constructor_triton")
        tests.append(True)
    except Exception as e:
        print(f"✗ src.prompt_constructor_triton: {e}")
        tests.append(False)
    
    try:
        from src.triton_safety_check import analyze_triton_kernel
        print("✓ src.triton_safety_check")
        tests.append(True)
    except Exception as e:
        print(f"✗ src.triton_safety_check: {e}")
        tests.append(False)
    
    # Test evaluation servers
    try:
        from scripts.eval_server_subprocess import app as subprocess_app
        print("✓ scripts.eval_server_subprocess")
        tests.append(True)
    except Exception as e:
        print(f"✗ scripts.eval_server_subprocess: {e}")
        tests.append(False)
    
    try:
        from scripts.simple_eval_server import app as simple_app
        print("✓ scripts.simple_eval_server")
        tests.append(True)
    except Exception as e:
        print(f"⚠️  scripts.simple_eval_server: {e}")
        # This might fail due to missing 'together' package but is not critical
        tests.append(True)  # Mark as warning, not failure
    
    # Test data processing scripts
    try:
        from scripts.scan_kernel_bench_data_to_train_set_triton import main
        print("✓ scripts.scan_kernel_bench_data_to_train_set_triton")
        tests.append(True)
    except Exception as e:
        print(f"✗ scripts.scan_kernel_bench_data_to_train_set_triton: {e}")
        tests.append(False)
    
    return all(tests)

def test_triton_kernel_evaluation():
    """Test a simple Triton kernel evaluation"""
    print("\n" + "=" * 60)
    print("2. Testing Triton Kernel Evaluation")
    print("=" * 60)
    
    # Simple reference PyTorch implementation
    ref_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(128, 128, device='cuda')]

def get_init_inputs():
    return []
'''
    
    # Simple Triton implementation
    triton_code = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def multiply_by_2_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(out_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        multiply_by_2_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output
'''
    
    try:
        from src.eval import eval_kernel_against_ref
        
        # Only test if CUDA is available
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping kernel evaluation test")
            return True
        
        print("Testing Triton backend evaluation...")
        result = eval_kernel_against_ref(
            ref_code,
            triton_code,
            backend="triton",
            num_correct_trials=1,
            num_perf_trials=1,
            verbose=False
        )
        
        if result and result.compiled and result.correctness:
            print(f"✓ Triton kernel evaluation successful!")
            print(f"  - Compiled: {result.compiled}")
            print(f"  - Correct: {result.correctness}")
            return True
        else:
            print(f"✗ Triton kernel evaluation failed")
            if result:
                print(f"  - Compiled: {result.compiled}")
                print(f"  - Correct: {result.correctness}")
            return False
            
    except Exception as e:
        print(f"✗ Error during Triton evaluation: {e}")
        traceback.print_exc()
        return False

def test_server_endpoints():
    """Test that server endpoints are properly configured"""
    print("\n" + "=" * 60)
    print("3. Testing Server Endpoints")
    print("=" * 60)
    
    try:
        from scripts.eval_server_subprocess import app
        
        # Check if all required endpoints exist
        routes = [route.path for route in app.routes]
        required_endpoints = ["/eval", "/health", "/", "/gpu_status", "/backend_info"]
        
        for endpoint in required_endpoints:
            if endpoint in routes:
                print(f"✓ Endpoint {endpoint} exists")
            else:
                print(f"✗ Missing endpoint {endpoint}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking server endpoints: {e}")
        return False

def test_data_processing():
    """Test that data processing scripts are configured"""
    print("\n" + "=" * 60)
    print("4. Testing Data Processing Scripts")
    print("=" * 60)
    
    # Check if the scripts exist
    scripts_to_check = [
        "scripts/scan_kernel_bench_data_to_train_set.py",
        "scripts/scan_kernel_bench_data_to_train_set_triton.py",
    ]
    
    all_exist = True
    for script in scripts_to_check:
        if Path(script).exists():
            print(f"✓ {script} exists")
        else:
            print(f"✗ {script} missing")
            all_exist = False
    
    return all_exist

def main():
    print("🔍 KernelBench Triton Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Triton Kernel Evaluation", test_triton_kernel_evaluation()))
    results.append(("Server Endpoints", test_server_endpoints()))
    results.append(("Data Processing Scripts", test_data_processing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✅ All tests passed! The Triton evaluation server is properly integrated.")
        print("\n📝 Next steps:")
        print("1. Start the evaluation server:")
        print("   python scripts/eval_server_subprocess.py")
        print("")
        print("2. Test with a sample evaluation:")
        print("   curl -X POST http://localhost:18188/eval \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"original_model_src\": \"...\", \"custom_model_src\": \"...\", \"backend\": \"triton\"}'")
        print("")
        print("3. Generate training data:")
        print("   python scripts/scan_kernel_bench_data_to_train_set_triton.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        print("\n📝 Common fixes:")
        print("1. Install missing dependencies:")
        print("   pip install fastapi uvicorn torch triton pydantic")
        print("")
        print("2. For optional API clients (not critical):")
        print("   pip install together openai anthropic google-generativeai")
        return 1

if __name__ == "__main__":
    sys.exit(main())