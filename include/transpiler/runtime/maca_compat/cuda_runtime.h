/* CUDA runtime API shim for MetaX mxcc (CUDA-compatible mc* runtime). */
#pragma once

#include "cuda_std_fwd.h"
#include <mcr/mc_runtime.h>

#ifndef cudaStream_t
typedef mcStream_t cudaStream_t;
#endif

#ifndef cudaError_t
typedef mcError_t cudaError_t;
#endif

#ifndef cudaSuccess
#define cudaSuccess mcSuccess
#endif

#ifndef cudaGetErrorString
#define cudaGetErrorString mcGetErrorString
#endif

#ifndef cudaMalloc
#define cudaMalloc mcMalloc
#endif

#ifndef cudaFree
#define cudaFree mcFree
#endif

#ifndef cudaMemcpy
#define cudaMemcpy mcMemcpy
#endif

#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync mcMemcpyAsync
#endif

#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost mcMemcpyDeviceToHost
#endif

#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice mcMemcpyHostToDevice
#endif

#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#endif

#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize mcStreamSynchronize
#endif

#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize mcDeviceSynchronize
#endif

#ifndef cudaGetLastError
#define cudaGetLastError mcGetLastError
#endif

#ifndef cudaFuncSetAttribute
// mxcc lacks NVCC kernel-name sugar; take address of __global__ stub.
#define cudaFuncSetAttribute(kernel, attr, val)                                  \
  mcFuncSetAttribute(reinterpret_cast<const void *>(&(kernel)), (attr), (val))
#endif

#ifndef cudaFuncAttribute
typedef mcFuncAttribute cudaFuncAttribute;
#endif

#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  mcFuncAttributeMaxDynamicSharedMemorySize
#endif
