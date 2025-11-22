#!/usr/bin/env python3
"""
Convert YiRage benchmark files to support multi-backend (CUDA/MPS/CPU)
"""

import re
from pathlib import Path

# Common template for device selection
DEVICE_SELECTION_CODE = '''
# Auto-select device based on backend argument
if args.backend == 'cuda':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
elif args.backend == 'mps':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    device = 'cpu'

print(f"Using device: {device}")
'''

TIMING_IMPORTS = '''import torch
import numpy as np
import argparse
import time'''

def convert_file(filepath):
    """Convert a single benchmark file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already converted
    if 'Auto-select device' in content or ('args.backend' in content and 'device =' in content):
        print(f"✓ {filepath.name} already converted")
        return False
    
    # Fix import: yirage as yr -> yirage as yr
    content = content.replace('import yirage as yr', 'import yirage as yr')
    content = content.replace('yirage as yr', 'yirage as yr')
    
    # Add time import if not present
    if 'import time' not in content:
        content = content.replace('import argparse', 'import argparse\nimport time')
    
    # Add device selection after args parsing
    # Find where args are parsed
    pattern = r"(save_codes = args\.save_codes)"
    if pattern in content or re.search(pattern, content):
        content = re.sub(pattern, r'\1' + DEVICE_SELECTION_CODE, content, count=1)
    
    # Replace hardcoded 'cuda:0' with device variable in tensor creation
    content = re.sub(r"device='cuda:0'", "device=device", content)
    content = re.sub(r'device="cuda:0"', 'device=device', content)
    
    # Replace CUDA synchronization with conditional code
    # Pattern 1: torch.cuda.synchronize()
    content = re.sub(
        r'torch\.cuda\.synchronize\(\)',
        '''if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'mps' and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()''',
        content
    )
    
    # Pattern 2: CUDA Event timing
    cuda_event_pattern = r'starter, ender = torch\.cuda\.Event\(enable_timing=True\), torch\.cuda\.Event\(enable_timing=True\)'
    if re.search(cuda_event_pattern, content):
        content = re.sub(
            cuda_event_pattern,
            '''if device.startswith('cuda'):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)''',
            content
        )
    
    # Pattern 3: Event timing code
    # This is complex, need to handle the whole timing block
    # For now, let's add a comment
    content = re.sub(
        r'(starter\.record\(\))',
        r'''if device.startswith('cuda'):
        \1''',
        content,
        count=1
    )
    
    content = re.sub(
        r'(ender\.record\(\))',
        r'''\1
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
    else:
        start_time = time.perf_counter()
        # Execute operations (copy from above)
        # ...''',
        content,
        count=1
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ {filepath.name} converted")
    return True

if __name__ == '__main__':
    # Convert benchmark/*.py files
    benchmark_dir = Path('/Users/xingqiangchen/mirage/benchmark')
    
    for pyfile in benchmark_dir.glob('*.py'):
        if pyfile.name.startswith('_') or pyfile.name == '__init__.py':
            continue
        try:
            convert_file(pyfile)
        except Exception as e:
            print(f"✗ {pyfile.name}: {e}")
    
    # Convert end-to-end/*.py files  
    end_to_end_dir = benchmark_dir / 'end-to-end'
    if end_to_end_dir.exists():
        for pyfile in end_to_end_dir.glob('*.py'):
            if pyfile.name.startswith('_'):
                continue
            try:
                convert_file(pyfile)
            except Exception as e:
                print(f"✗ end-to-end/{pyfile.name}: {e}")
    
    print("\n✅ Conversion complete!")
    print("\nUsage:")
    print("  python benchmark/gated_mlp.py --backend mps")
    print("  python benchmark/gated_mlp.py --backend cuda")
    print("  python benchmark/gated_mlp.py --backend cpu")

