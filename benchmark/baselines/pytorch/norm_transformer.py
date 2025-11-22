import torch
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
parser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'],
                    help='Backend device: cuda, mps, or cpu')
args = parser.parse_args()
print("Batch size", args.batch)
print("Backend", args.backend)

# Select device
if args.backend == 'cuda' and torch.cuda.is_available():
    device = 'cuda'
elif args.backend == 'mps' and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

X = torch.rand([2 * args.batch, 4096], dtype=torch.float16, device=device)
W = torch.rand([4096, 4096], dtype=torch.float16, device=device)
rms_norm64 = torch.nn.RMSNorm(4096, device=device, dtype=torch.float16)

if device == 'cuda':
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
    S = rms_norm64(X)
    S = torch.matmul(S, W)

with torch.no_grad():
  for rep in range(repetitions):
      if device == 'cuda':
          starter.record()
          S = rms_norm64(X)
          O = torch.matmul(S, W)
          ender.record()
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
      else:
          start = time.perf_counter()
          S = rms_norm64(X)
          O = torch.matmul(S, W)
          if device == 'mps' and hasattr(torch.mps, 'synchronize'):
              torch.mps.synchronize()
          end = time.perf_counter()
          curr_time = (end - start) * 1000
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)
