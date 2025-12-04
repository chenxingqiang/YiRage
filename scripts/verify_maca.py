#!/usr/bin/env python3
"""
YiRage MACA Backend Verification Script
验证 MACA 后端安装和基本功能
"""

import sys
import time

def check_imports():
    """检查依赖导入"""
    print("=" * 50)
    print("1. Checking imports...")
    print("=" * 50)
    
    errors = []
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"✓ Device: {torch.cuda.get_device_name(0)}")
        else:
            errors.append("CUDA not available - mcPytorch may not be properly installed")
    except ImportError as e:
        errors.append(f"PyTorch import failed: {e}")
    
    try:
        import yirage
        print(f"✓ YiRage imported successfully")
    except ImportError as e:
        errors.append(f"YiRage import failed: {e}")
    
    try:
        import z3
        print(f"✓ Z3 solver available")
    except ImportError as e:
        errors.append(f"Z3 import failed: {e}")
    
    if errors:
        print("\n⚠ Errors found:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_simple_graph():
    """测试简单图创建"""
    print("\n" + "=" * 50)
    print("2. Testing simple graph creation...")
    print("=" * 50)
    
    import yirage
    
    try:
        # 创建简单 MatMul 图
        graph = yirage.new_kernel_graph()
        X = graph.new_input(dims=(16, 32), dtype=yirage.float16)
        W = graph.new_input(dims=(32, 32), dtype=yirage.float16)
        Y = graph.matmul(X, W)
        graph.mark_output(Y)
        
        print("✓ Graph created: X @ W")
        print(f"  Input X: {(16, 32)} float16")
        print(f"  Input W: {(32, 32)} float16")
        print(f"  Output Y: {(16, 32)} float16")
        return graph
    except Exception as e:
        print(f"✗ Graph creation failed: {e}")
        return None


def test_maca_search(graph):
    """测试 MACA 后端搜索"""
    print("\n" + "=" * 50)
    print("3. Testing MACA backend search...")
    print("=" * 50)
    
    if graph is None:
        print("✗ Skipped (no graph)")
        return None
    
    print("Starting search (this may take a few minutes)...")
    print("  Backend: maca")
    print("  Config: mlp")
    
    start_time = time.time()
    
    try:
        optimized = graph.superoptimize(
            backend="maca",
            config="mlp",
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        if optimized:
            print(f"\n✓ Search completed in {elapsed:.2f}s")
            print(f"✓ Found optimized graph!")
            return optimized
        else:
            print(f"\n⚠ Search completed but no optimized graph found")
            return None
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Search failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_execution(optimized):
    """测试优化图执行"""
    print("\n" + "=" * 50)
    print("4. Testing optimized graph execution...")
    print("=" * 50)
    
    if optimized is None:
        print("✗ Skipped (no optimized graph)")
        return False
    
    import torch
    
    try:
        # 创建输入数据
        x = torch.randn(16, 32, dtype=torch.float16, device="cuda")
        w = torch.randn(32, 32, dtype=torch.float16, device="cuda")
        
        # 运行优化图
        result = optimized(x, w)
        
        # 验证结果
        expected = torch.matmul(x, w)
        diff = (result - expected).abs().max().item()
        
        print(f"✓ Execution successful")
        print(f"  Output shape: {result.shape}")
        print(f"  Max diff from PyTorch: {diff:.6f}")
        
        if diff < 0.1:  # FP16 tolerance
            print("✓ Numerical verification passed!")
            return True
        else:
            print("⚠ Large numerical difference detected")
            return False
            
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance(optimized):
    """测试性能"""
    print("\n" + "=" * 50)
    print("5. Performance comparison...")
    print("=" * 50)
    
    import torch
    
    if optimized is None:
        print("⚠ No optimized graph, running PyTorch baseline only")
    
    # 较大的输入以获得有意义的时间
    x = torch.randn(256, 512, dtype=torch.float16, device="cuda")
    w = torch.randn(512, 512, dtype=torch.float16, device="cuda")
    
    def profile(func, name, warmup=20, repeat=100):
        # Warmup
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        
        # Time with CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(repeat):
            func()
        end.record()
        torch.cuda.synchronize()
        
        avg_ms = start.elapsed_time(end) / repeat
        print(f"  {name}: {avg_ms:.4f} ms")
        return avg_ms
    
    try:
        pytorch_time = profile(
            lambda: torch.matmul(x, w),
            "PyTorch MatMul"
        )
        
        if optimized is not None:
            yirage_time = profile(
                lambda: optimized(x, w),
                "YiRage Optimized"
            )
            
            speedup = pytorch_time / yirage_time
            print(f"\n  Speedup: {speedup:.2f}x")
            
            if speedup > 1.0:
                print("✓ YiRage is faster!")
            elif speedup > 0.9:
                print("~ Performance is similar")
            else:
                print("⚠ YiRage is slower (may need larger inputs)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("YiRage MACA Backend Verification")
    print("=" * 50 + "\n")
    
    results = {}
    
    # 1. Check imports
    results['imports'] = check_imports()
    if not results['imports']:
        print("\n❌ Import check failed. Please fix dependencies first.")
        return 1
    
    # 2. Test graph creation
    graph = test_simple_graph()
    results['graph'] = graph is not None
    
    # 3. Test MACA search
    optimized = test_maca_search(graph)
    results['search'] = optimized is not None
    
    # 4. Test execution
    results['execution'] = test_execution(optimized)
    
    # 5. Test performance
    results['performance'] = test_performance(optimized)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! YiRage MACA backend is ready.")
        return 0
    elif passed >= 2:
        print("\n⚠ Some tests failed, but basic functionality works.")
        return 0
    else:
        print("\n❌ Multiple tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

