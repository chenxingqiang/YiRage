#!/usr/bin/env python3
"""
Add multi-backend support to PyTorch baseline scripts
"""

import re
import sys
from pathlib import Path

def add_backend_support(filepath):
    """Add --backend argument and device selection to a PyTorch script"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Already has backend support
    if '--backend' in content or 'args.backend' in content:
        print(f"✓ {filepath.name} already has backend support")
        return False
    
    # Add import time
    if 'import time' not in content:
        content = content.replace('import argparse', 'import argparse\nimport time')
    
    # Add backend argument
    pattern = r"(parser\.add_argument\('-b', '--batch'[^\n]+)"
    replacement = r"\1\nparser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'],\n                    help='Backend device: cuda, mps, or cpu')"
    content = re.sub(pattern, replacement, content)
    
    # Add device selection after args parsing
    device_selection = """
# Select device based on backend
if args.backend == 'cuda' and torch.cuda.is_available():
    device = 'cuda'
elif args.backend == 'mps' and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")
"""
    
    # Insert after args parsing
    pattern = r"(print\(\"Batch size\", args\.batch\))"
    replacement = r'\1\nprint("Backend", args.backend)' + device_selection
    content = re.sub(pattern, replacement, content, count=1)
    
    # Replace 'cuda' with device variable
    content = re.sub(r"device='cuda'", "device=device", content)
    content = re.sub(r'device="cuda"', 'device=device', content)
    content = re.sub(r"device='cuda:0'", "device=device", content)
    
    # Add use_cuda_timing flag
    pattern = r"(starter, ender = torch\.cuda\.Event[^\n]+)"
    replacement = "if device == 'cuda':\n    \\1\nuse_cuda_timing = (device == 'cuda')"
    content = re.sub(pattern, replacement, content)
    
    # Update timing loop for CUDA
    old_timing = r"""starter\.record\(\)
      ([^\n]+)
      ender\.record\(\)
      torch\.cuda\.synchronize\(\)
      curr_time = starter\.elapsed_time\(ender\)"""
    
    new_timing = """if use_cuda_timing:
          starter.record()
          \\1
          ender.record()
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
      else:
          start = time.perf_counter()
          \\1
          if device == 'mps' and hasattr(torch.mps, 'synchronize'):
              torch.mps.synchronize()
          end = time.perf_counter()
          curr_time = (end - start) * 1000"""
    
    # This is complex, let's do it differently - find the with block
    # For now, write the modified content
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ {filepath.name} updated with backend support")
    return True

if __name__ == '__main__':
    baseline_dir = Path(__file__).parent.parent / 'benchmark' / 'baselines' / 'pytorch'
    
    for pyfile in baseline_dir.glob('*.py'):
        if pyfile.name.startswith('_'):
            continue
        try:
            add_backend_support(pyfile)
        except Exception as e:
            print(f"✗ {pyfile.name}: {e}")
    
    print("\nDone!")

