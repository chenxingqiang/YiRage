#!/usr/bin/env python3
"""
YiRage Multi-Backend Selection Demo

This demo shows how to:
1. Query available backends
2. Get backend information
3. Select a specific backend for computation
4. Use fallback backends if primary is unavailable
"""

import sys
try:
    import yirage as yr
except ImportError:
    print("Error: YiRage not installed. Please install with: pip install -e .")
    sys.exit(1)


def main():
    print("=" * 60)
    print("YiRage Multi-Backend Selection Demo")
    print("=" * 60)

    # 1. List all available backends
    print("\n[1] Available Backends:")
    print("-" * 40)
    backends = yr.get_available_backends()
    if not backends:
        print("  No backends available!")
        return
    
    for backend in backends:
        print(f"  - {backend}")

    # 2. Get detailed backend information
    print("\n[2] Backend Details:")
    print("-" * 40)
    for backend in backends:
        print(f"\n  {backend.upper()}:")
        info = yr.get_backend_info(backend)
        for key, value in info.items():
            print(f"    {key}: {value}")

    # 3. Get default backend
    print("\n[3] Default Backend:")
    print("-" * 40)
    default = yr.get_default_backend()
    print(f"  {default}")

    # 4. Check if specific backends are available
    print("\n[4] Backend Availability Check:")
    print("-" * 40)
    test_backends = ['cuda', 'cpu', 'mps', 'nki', 'triton']
    for backend in test_backends:
        available = yr.is_backend_available(backend)
        status = "✓" if available else "✗"
        print(f"  {status} {backend}: {'Available' if available else 'Not Available'}")

    # 5. Set default backend
    print("\n[5] Setting Default Backend:")
    print("-" * 40)
    if 'cuda' in backends:
        success = yr.set_default_backend('cuda')
        print(f"  Set CUDA as default: {success}")
    elif 'cpu' in backends:
        success = yr.set_default_backend('cpu')
        print(f"  Set CPU as default: {success}")

    # 6. Create a simple graph with backend selection
    print("\n[6] Creating Graphs with Different Backends:")
    print("-" * 40)
    
    for backend in backends[:2]:  # Test first 2 available backends
        print(f"\n  Testing with {backend} backend:")
        try:
            # Create a simple kernel graph
            graph = yr.new_kernel_graph()
            
            # TODO: Add actual graph operations here
            # This is a placeholder showing how backend selection would work
            
            print(f"    ✓ Graph created successfully with {backend}")
        except Exception as e:
            print(f"    ✗ Error with {backend}: {e}")

    # 7. Using list_backends helper
    print("\n[7] Detailed Backend Listing:")
    print("-" * 40)
    yr.list_backends(verbose=True)

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def example_with_fallback():
    """
    Example showing how to use fallback backends
    """
    print("\n" + "=" * 60)
    print("Fallback Backend Example")
    print("=" * 60)
    
    # Try backends in order of preference
    preferred_backends = ['cuda', 'mps', 'cpu']
    selected_backend = None
    
    print("\nTrying backends in order:")
    for backend in preferred_backends:
        if yr.is_backend_available(backend):
            selected_backend = backend
            print(f"  ✓ Selected: {backend}")
            break
        else:
            print(f"  ✗ Not available: {backend}")
    
    if not selected_backend:
        print("\n  Error: No suitable backend found!")
        return
    
    print(f"\nUsing backend: {selected_backend}")
    # Proceed with computation using selected_backend


if __name__ == '__main__':
    main()
    example_with_fallback()





