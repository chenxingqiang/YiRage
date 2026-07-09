/* cuBLAS v2 API shim for MetaX mcblas (transpiler runtime matmul). */
#pragma once

#include "mc_library_types.h"
#include "mcblas.h"

typedef mcblasHandle_t cublasHandle_t;
typedef mcblasStatus_t cublasStatus_t;
typedef mcblasOperation_t cublasOperation_t;
typedef mcblasComputeType_t cublasComputeType_t;
typedef mcblasGemmAlgo_t cublasGemmAlgo_t;
typedef macaDataType cudaDataType_t;

#define CUBLAS_STATUS_SUCCESS MCBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N MCBLAS_OP_N
#define CUBLAS_OP_T MCBLAS_OP_T
#define CUBLAS_COMPUTE_16F MCBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_16F_PEDANTIC MCBLAS_COMPUTE_16F_PEDANTIC
#define CUBLAS_COMPUTE_32F MCBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_PEDANTIC MCBLAS_COMPUTE_32F_PEDANTIC
#define CUBLAS_GEMM_DEFAULT MCBLAS_GEMM_DEFAULT

#define CUDA_R_16F MACA_R_16F
#define CUDA_R_16BF MACA_R_16BF
#define CUDA_R_32F MACA_R_32F

#define cublasCreate mcblasCreate
#define cublasDestroy mcblasDestroy
#define cublasGemmStridedBatchedEx mcblasGemmStridedBatchedEx
#define cublasGetStatusName mcblasGetStatusName
#define cublasGetStatusString mcblasGetStatusString
