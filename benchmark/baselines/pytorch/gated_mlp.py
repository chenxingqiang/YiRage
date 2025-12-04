import torch
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
parser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu', 'maca'],
                    help='Backend device: cuda, mps, cpu, or maca')
args = parser.parse_args()
print("Batch size", args.batch)
print("Backend", args.backend)

# Select device based on backend argument
# MACA uses CUDA interface via mcPytorch
if args.backend == 'maca' and torch.cuda.is_available():
    device = 'cuda'  # MACA maps to CUDA device
elif args.backend == 'cuda' and torch.cuda.is_available():
    device = 'cuda'
elif args.backend == 'mps' and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

silu = torch.nn.SiLU()

X = torch.rand([args.batch, 4096], dtype=torch.float16, device=device)
A = torch.rand([4096, 4096], dtype=torch.float16, device=device)
B = torch.rand([4096, 4096], dtype=torch.float16, device=device)

# Create timing events based on backend
if device == 'cuda':
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
use_cuda_timing = (device == 'cuda')

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(128):
      O1 = torch.matmul(X, A)
      O2 = torch.matmul(X, B)
      O1 = silu(O1)
      O = torch.mul(O1, O2)

with torch.no_grad():
  for rep in range(repetitions):
      if use_cuda_timing:
          starter.record()
          O1 = torch.matmul(X, A)
          O2 = torch.matmul(X, B)
          O1 = silu(O1)
          O = torch.mul(O1, O2)
          ender.record()
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
      else:
          # Use Python timing for MPS/CPU
          start = time.perf_counter()
          O1 = torch.matmul(X, A)
          O2 = torch.matmul(X, B)
          O1 = silu(O1)
          O = torch.mul(O1, O2)
          if device == 'mps' and hasattr(torch.mps, 'synchronize'):
              torch.mps.synchronize()
          end = time.perf_counter()
          curr_time = (end - start) * 1000  # Convert to ms
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

