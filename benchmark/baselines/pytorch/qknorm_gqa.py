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

Q = torch.rand([args.batch, 2, 256, 64], dtype=torch.float16, device=device)
K = torch.rand([args.batch, 2, 64, 4096], dtype=torch.float16, device=device)
V = torch.rand([args.batch, 2, 4096, 64], dtype=torch.float16, device=device)

multihead_attn = torch.nn.MultiheadAttention(embed_dim=32 * 64, num_heads = 2, batch_first=True, device=device, dtype=torch.float16)

if device == 'cuda':
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
rms_norm64 = torch.nn.RMSNorm(64, device=device, dtype=torch.float16)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
    Q = rms_norm64(Q)
    V = rms_norm64(V)
    S = torch.matmul(Q, K)
    S = torch.softmax(S, dim=3)
    S = torch.matmul(S, V)

with torch.no_grad():
  for rep in range(repetitions):
      if device == 'cuda':
          starter.record()
          Q = rms_norm64(Q)
          V = rms_norm64(V)
          S = torch.matmul(Q, K)
          S = torch.softmax(S, dim=3)
          S = torch.matmul(S, V)
          ender.record()
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
      else:
          start = time.perf_counter()
          Q = rms_norm64(Q)
          V = rms_norm64(V)
          S = torch.matmul(Q, K)
          S = torch.softmax(S, dim=3)
          S = torch.matmul(S, V)
          if device == 'mps' and hasattr(torch.mps, 'synchronize'):
              torch.mps.synchronize()
          end = time.perf_counter()
          curr_time = (end - start) * 1000
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)
