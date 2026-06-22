#ifndef YIRAGE_USE_CUTLASS_KERNEL
// YiRage use a flag (use_cutlass_kernel) to control whether use the cutlass
// version kernel or not.
#define YIRAGE_USE_CUTLASS_KERNEL 1
#endif // YIRAGE_USE_CUTLASS_KERNEL

#include "argmax.cuh"
#include "embedding.cuh"
#include "multitoken_paged_attention.cuh"
#include "reduction.cuh"
#include "rmsnorm.cuh"
#include "rotary_embedding.cuh"
#include "silu_mul.cuh"
#include "identity.cuh"

#if YIRAGE_USE_CUTLASS_KERNEL
#include "linear_cutlass.cuh"
#else
#include "linear.cuh"
#endif // YIRAGE_USE_CUTLASS_KERNEL