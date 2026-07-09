/* cuComplex.h shim for MetaX mxcc (CUTLASS complex types). */
#pragma once

#include "mcComplex.h"

typedef mcFloatComplex cuFloatComplex;
typedef mcDoubleComplex cuDoubleComplex;
typedef mcFloatComplex cuComplex;

#define cuCrealf mcCrealf
#define cuCimagf mcCimagf
#define cuCreal mcCreal
#define cuCimag mcCimag
#define make_cuFloatComplex make_mcFloatComplex
#define make_cuDoubleComplex make_mcDoubleComplex
#define cuConjf mcConjf
#define cuConj mcConj
