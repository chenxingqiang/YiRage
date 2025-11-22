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

X = torch.rand([args.batch, 4096], dtype=torch.float16, device=device)
A = torch.rand([4096, 512], dtype=torch.float16, device=device)
B = torch.rand([512, 4096], dtype=torch.float16, device=device)

if device == 'cuda':
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(128):
    O1 = torch.matmul(X, A)
    O = torch.matmul(O1, B)

with torch.no_grad():
  for rep in range(repetitions):
      if device == 'cuda':
          starter.record()
          O1 = torch.matmul(X, A)
          O = torch.matmul(O1, B)
          ender.record()
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
      else:
          start = time.perf_counter()
          O1 = torch.matmul(X, A)
          O = torch.matmul(O1, B)
          if device == 'mps' and hasattr(torch.mps, 'synchronize'):
              torch.mps.synchronize()
          end = time.perf_counter()
          curr_time = (end - start) * 1000
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)
