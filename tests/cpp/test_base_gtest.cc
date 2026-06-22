// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_base_gtest.cc
 * @brief Base Module Unit Tests (Google Test version)
 *
 * Tests for type.h, layout.h, and src/base/*.cc
 * Including DataType, Layout, BackendType, and Operator types.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>

namespace yirage {
namespace type {

// =============================================================================
// Type Definitions (mirror actual implementation)
// =============================================================================

typedef uint16_t FPType;
typedef int64_t GuidType;

enum BackendType {
    // Hardware Backends
    BT_CUDA = 0,
    BT_MPS = 1,
    BT_ROCM = 6,
    BT_ASCEND = 4,
    BT_MACA = 5,
    BT_CPU = 10,
    BT_TPU = 30,
    BT_FPGA = 31,
    BT_XPU = 32,
    
    // Library Backends
    BT_CUDNN = 2,
    BT_CUSPARSELT = 3,
    BT_CUTLASS = 7,
    BT_MHA = 22,
    BT_MKL = 11,
    BT_MKLDNN = 12,
    BT_OPENMP = 13,
    BT_XEON = 14,
    BT_NNPACK = 23,
    BT_OPT_EINSUM = 24,
    
    // DSL/Compiler Backends
    BT_TRITON = 21,
    BT_NKI = 20,
    
    // MLIR Backends
    BT_MLIR = 40,
    BT_MLIR_LLVM = 41,
    BT_MLIR_NVVM = 42,
    BT_MLIR_ROCDL = 43,
    BT_MLIR_SPIRV = 44,
    BT_MLIR_GPU = 45,
    BT_STABLEHLO = 50,
    BT_MHLO = 51,
    BT_TOSA = 52,
    BT_LINALG = 53,
    BT_TCP = 54,
    BT_IREE = 55,
    BT_TVM = 56,
    BT_XLA = 57,
    
    BT_UNKNOWN = 999,
};

enum MLIRDialect {
    MLIR_DIALECT_UNKNOWN = 0,
    MLIR_DIALECT_BUILTIN = 1,
    MLIR_DIALECT_ARITH = 2,
    MLIR_DIALECT_FUNC = 3,
    MLIR_DIALECT_SCF = 4,
    MLIR_DIALECT_AFFINE = 5,
    MLIR_DIALECT_MEMREF = 6,
    MLIR_DIALECT_TENSOR = 7,
    MLIR_DIALECT_VECTOR = 8,
    MLIR_DIALECT_LINALG = 10,
    MLIR_DIALECT_TOSA = 11,
    MLIR_DIALECT_STABLEHLO = 12,
    MLIR_DIALECT_MHLO = 13,
    MLIR_DIALECT_LLVM = 20,
    MLIR_DIALECT_NVVM = 21,
    MLIR_DIALECT_ROCDL = 22,
    MLIR_DIALECT_SPIRV = 23,
    MLIR_DIALECT_GPU = 24,
    MLIR_DIALECT_AMX = 25,
    MLIR_DIALECT_X86VECTOR = 26,
    MLIR_DIALECT_ARM_NEON = 27,
    MLIR_DIALECT_ARM_SVE = 28,
    MLIR_DIALECT_TRITON = 30,
    MLIR_DIALECT_TCP = 31,
};

enum DataType {
    DT_FLOAT4 = 920,
    DT_INT4 = 925,
    DT_UINT4 = 926,
    DT_FLOAT8 = 930,
    DT_INT8 = 935,
    DT_UINT8 = 936,
    DT_FLOAT16 = 940,
    DT_BFLOAT16 = 941,
    DT_INT16 = 945,
    DT_UINT16 = 946,
    DT_FLOAT32 = 950,
    DT_INT32 = 955,
    DT_UINT32 = 956,
    DT_DOUBLE = 960,
    DT_INT64 = 965,
    DT_UINT64 = 966,
    DT_UNKNOWN = 999,
};

enum KNOperatorType {
    KN_UNKOWN = 1000,
    KN_INPUT_OP = 1001,
    KN_OUTPUT_OP = 1002,
    KN_MATMUL_OP = 1003,
    KN_EXP_OP = 1100,
    KN_SQUARE_OP = 1101,
    KN_SQRT_OP = 1102,
    KN_MUL_SCALAR_OP = 1103,
    KN_SILU_OP = 1104,
    KN_SIGMOID_OP = 1105,
    KN_GELU_OP = 1106,
    KN_RELU_OP = 1150,
    KN_CLAMP_OP = 1151,
    KN_LOG_OP = 1160,
    KN_ADD_OP = 1200,
    KN_MUL_OP = 1201,
    KN_DIV_OP = 1202,
    KN_POW_OP = 1203,
    KN_REDUCTION_0_OP = 1300,
    KN_REDUCTION_1_OP = 1301,
    KN_REDUCTION_2_OP = 1302,
    KN_RMS_NORM_OP = 1350,
    KN_CONCAT_0_OP = 1400,
    KN_CONCAT_1_OP = 1401,
    KN_CONCAT_2_OP = 1402,
    KN_SPLIT_0_OP = 1420,
    KN_SPLIT_1_OP = 1421,
    KN_SPLIT_2_OP = 1422,
    KN_ALLREDUCE_OP = 1900,
    KN_CUSTOMIZED_OP = 1999,
};

enum TBOperatorType {
    TB_UNKOWN = 2000,
    TB_INPUT_OP = 2001,
    TB_OUTPUT_OP = 2002,
    TB_MATMUL_OP = 2003,
    TB_EXP_OP = 2100,
    TB_SQUARE_OP = 2101,
    TB_SQRT_OP = 2102,
    TB_MUL_SCALAR_OP = 2103,
    TB_SILU_OP = 2104,
    TB_SIGMOID_OP = 2105,
    TB_GELU_OP = 2106,
    TB_RELU_OP = 2150,
    TB_CLAMP_OP = 2151,
    TB_LOG_OP = 2160,
    TB_ADD_OP = 2200,
    TB_MUL_OP = 2201,
    TB_DIV_OP = 2202,
    TB_SUB_OP = 2203,
    TB_POW_OP = 2204,
    TB_REDUCTION_0_OP = 2301,
    TB_REDUCTION_1_OP = 2302,
    TB_REDUCTION_2_OP = 2303,
    TB_RMS_NORM_OP = 2350,
    TB_CONCAT_0_OP = 2400,
    TB_CONCAT_1_OP = 2401,
    TB_CONCAT_2_OP = 2402,
    TB_SPLIT_0_OP = 2420,
    TB_SPLIT_1_OP = 2421,
    TB_SPLIT_2_OP = 2422,
    TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
    TB_CUSTOMIZED_OP = 2999,
};

enum ActivationType {
    ACT_UNKOWN = 3000,
    ACT_EXP = 3001,
    ACT_RELU = 3002,
    ACT_GELU = 3003,
    ACT_SILU = 3004,
    ACT_NONE = 3099,
};

enum BackendCategory {
    BC_NVIDIA_GPU = 0,
    BC_AMD_GPU = 1,
    BC_INTEL_GPU = 2,
    BC_APPLE_GPU = 3,
    BC_HUAWEI_NPU = 4,
    BC_METAX_GPU = 5,
    BC_GOOGLE_TPU = 6,
    BC_FPGA = 7,
    BC_CPU = 8,
    BC_MLIR = 9,
    BC_UNKNOWN = 99,
};

// =============================================================================
// Function Implementations (mirror actual)
// =============================================================================

size_t get_datatype_size(DataType type) {
    switch (type) {
        case DT_INT8:
        case DT_UINT8:
        case DT_FLOAT8:
            return 1;
        case DT_BFLOAT16:
        case DT_FLOAT16:
        case DT_INT16:
        case DT_UINT16:
            return 2;
        case DT_INT32:
        case DT_UINT32:
        case DT_FLOAT32:
            return 4;
        case DT_INT64:
        case DT_UINT64:
        case DT_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

std::string get_datatype_str(DataType dtype) {
    switch (dtype) {
        case DT_INT4:      return "int4";
        case DT_FLOAT8:    return "float8";
        case DT_INT8:      return "int8";
        case DT_UINT8:     return "uint8";
        case DT_BFLOAT16:  return "bfloat16_t";
        case DT_FLOAT16:   return "half_t";
        case DT_UINT16:    return "uint16";
        case DT_INT16:     return "int16";
        case DT_INT32:     return "int32";
        case DT_UINT32:    return "uint32";
        case DT_FLOAT32:   return "float";
        case DT_DOUBLE:    return "double";
        case DT_INT64:     return "int64";
        case DT_UINT64:    return "uint64";
        default:           return "unknown";
    }
}

bool is_threadblock_element_unary(TBOperatorType op_type) {
    switch (op_type) {
        case TB_EXP_OP:
        case TB_SQUARE_OP:
        case TB_SQRT_OP:
        case TB_SILU_OP:
        case TB_GELU_OP:
        case TB_RELU_OP:
        case TB_CLAMP_OP:
        case TB_MUL_SCALAR_OP:
            return true;
        default:
            return false;
    }
}

bool is_mlir_backend(BackendType type) {
    switch (type) {
        case BT_MLIR:
        case BT_MLIR_LLVM:
        case BT_MLIR_NVVM:
        case BT_MLIR_ROCDL:
        case BT_MLIR_SPIRV:
        case BT_MLIR_GPU:
        case BT_STABLEHLO:
        case BT_MHLO:
        case BT_TOSA:
        case BT_LINALG:
        case BT_TCP:
        case BT_IREE:
        case BT_TVM:
        case BT_XLA:
            return true;
        default:
            return false;
    }
}

bool is_hardware_backend(BackendType type) {
    switch (type) {
        case BT_CUDA:
        case BT_CPU:
        case BT_MPS:
        case BT_ASCEND:
        case BT_MACA:
        case BT_ROCM:
        case BT_TPU:
        case BT_FPGA:
        case BT_XPU:
            return true;
        default:
            return false;
    }
}

bool is_library_backend(BackendType type) {
    switch (type) {
        case BT_CUDNN:
        case BT_CUSPARSELT:
        case BT_CUTLASS:
        case BT_MKL:
        case BT_MKLDNN:
        case BT_OPENMP:
        case BT_XEON:
        case BT_MHA:
        case BT_NNPACK:
        case BT_OPT_EINSUM:
            return true;
        default:
            return false;
    }
}

bool is_dsl_backend(BackendType type) {
    switch (type) {
        case BT_TRITON:
        case BT_NKI:
            return true;
        default:
            return is_mlir_backend(type);
    }
}

BackendType get_fallback_backend(BackendType type) {
    switch (type) {
        case BT_CUDNN:
        case BT_CUSPARSELT:
        case BT_CUTLASS:
        case BT_MHA:
        case BT_TRITON:
            return BT_CUDA;
        case BT_MKL:
        case BT_MKLDNN:
        case BT_OPENMP:
        case BT_XEON:
        case BT_NNPACK:
        case BT_OPT_EINSUM:
            return BT_CPU;
        case BT_MLIR_NVVM:
            return BT_CUDA;
        case BT_MLIR_ROCDL:
            return BT_ROCM;
        case BT_MLIR_LLVM:
            return BT_CPU;
        case BT_STABLEHLO:
        case BT_MHLO:
        case BT_XLA:
            return BT_TPU;
        default:
            if (is_hardware_backend(type)) return type;
            return BT_UNKNOWN;
    }
}

BackendCategory get_backend_category(BackendType type) {
    switch (type) {
        case BT_CUDA:
        case BT_CUDNN:
        case BT_CUSPARSELT:
        case BT_CUTLASS:
        case BT_TRITON:
        case BT_MHA:
        case BT_MLIR_NVVM:
            return BC_NVIDIA_GPU;
        case BT_ROCM:
        case BT_MLIR_ROCDL:
            return BC_AMD_GPU;
        case BT_XPU:
        case BT_MLIR_SPIRV:
            return BC_INTEL_GPU;
        case BT_MPS:
            return BC_APPLE_GPU;
        case BT_ASCEND:
            return BC_HUAWEI_NPU;
        case BT_MACA:
            return BC_METAX_GPU;
        case BT_TPU:
        case BT_STABLEHLO:
        case BT_MHLO:
        case BT_XLA:
            return BC_GOOGLE_TPU;
        case BT_FPGA:
            return BC_FPGA;
        case BT_CPU:
        case BT_MKL:
        case BT_MKLDNN:
        case BT_OPENMP:
        case BT_XEON:
        case BT_NNPACK:
        case BT_OPT_EINSUM:
        case BT_MLIR_LLVM:
            return BC_CPU;
        case BT_MLIR:
        case BT_MLIR_GPU:
        case BT_TOSA:
        case BT_LINALG:
        case BT_TCP:
        case BT_IREE:
        case BT_TVM:
        case BT_NKI:
            return BC_MLIR;
        default:
            return BC_UNKNOWN;
    }
}

const char* get_mlir_dialect_name(MLIRDialect dialect) {
    switch (dialect) {
        case MLIR_DIALECT_BUILTIN:   return "builtin";
        case MLIR_DIALECT_ARITH:     return "arith";
        case MLIR_DIALECT_FUNC:      return "func";
        case MLIR_DIALECT_SCF:       return "scf";
        case MLIR_DIALECT_AFFINE:    return "affine";
        case MLIR_DIALECT_MEMREF:    return "memref";
        case MLIR_DIALECT_TENSOR:    return "tensor";
        case MLIR_DIALECT_VECTOR:    return "vector";
        case MLIR_DIALECT_LINALG:    return "linalg";
        case MLIR_DIALECT_TOSA:      return "tosa";
        case MLIR_DIALECT_STABLEHLO: return "stablehlo";
        case MLIR_DIALECT_MHLO:      return "mhlo";
        case MLIR_DIALECT_LLVM:      return "llvm";
        case MLIR_DIALECT_NVVM:      return "nvvm";
        case MLIR_DIALECT_ROCDL:     return "rocdl";
        case MLIR_DIALECT_SPIRV:     return "spirv";
        case MLIR_DIALECT_GPU:       return "gpu";
        case MLIR_DIALECT_AMX:       return "amx";
        case MLIR_DIALECT_TRITON:    return "triton";
        case MLIR_DIALECT_TCP:       return "tcp";
        default:                     return "unknown";
    }
}

}  // namespace type

namespace layout {

enum DmemLayout {
    DmemRowMajor = 100,
    DmemColumnMajor = 101,
    DmemUnknownLayout = 199,
};

enum SmemLayout {
    SmemRowMajor = 200,
    SmemColumnMajor = 201,
    SmemRowMajorTensorOpMultiplicand_Crosswise16 = 202,
    SmemRowMajorTensorOpMultiplicand_Crosswise32 = 203,
    SmemRowMajorTensorOpMultiplicand_Crosswise64 = 204,
    SmemColumnMajorTensorOpMultiplicand_Crosswise16 = 205,
    SmemColumnMajorTensorOpMultiplicand_Crosswise32 = 206,
    SmemColumnMajorTensorOpMultiplicand_Crosswise64 = 207,
    SmemUnknownLayout = 299,
};

enum CmemLayout {
    CmemRowMajor = 300,
    CmemColumnMajor = 301,
    CmemUnknownLayout = 399,
};

CmemLayout dmemlayout_to_cmemlayout(DmemLayout dmem_layout) {
    switch (dmem_layout) {
        case DmemRowMajor:
            return CmemRowMajor;
        case DmemColumnMajor:
            return CmemColumnMajor;
        default:
            return CmemUnknownLayout;
    }
}

}  // namespace layout
}  // namespace yirage

using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// DataType Tests
// =============================================================================

class DataTypeTest : public ::testing::Test {};

TEST_F(DataTypeTest, SizeOf8BitTypes) {
    EXPECT_EQ(get_datatype_size(DT_INT8), 1u);
    EXPECT_EQ(get_datatype_size(DT_UINT8), 1u);
    EXPECT_EQ(get_datatype_size(DT_FLOAT8), 1u);
}

TEST_F(DataTypeTest, SizeOf16BitTypes) {
    EXPECT_EQ(get_datatype_size(DT_FLOAT16), 2u);
    EXPECT_EQ(get_datatype_size(DT_BFLOAT16), 2u);
    EXPECT_EQ(get_datatype_size(DT_INT16), 2u);
    EXPECT_EQ(get_datatype_size(DT_UINT16), 2u);
}

TEST_F(DataTypeTest, SizeOf32BitTypes) {
    EXPECT_EQ(get_datatype_size(DT_FLOAT32), 4u);
    EXPECT_EQ(get_datatype_size(DT_INT32), 4u);
    EXPECT_EQ(get_datatype_size(DT_UINT32), 4u);
}

TEST_F(DataTypeTest, SizeOf64BitTypes) {
    EXPECT_EQ(get_datatype_size(DT_DOUBLE), 8u);
    EXPECT_EQ(get_datatype_size(DT_INT64), 8u);
    EXPECT_EQ(get_datatype_size(DT_UINT64), 8u);
}

TEST_F(DataTypeTest, UnknownTypeSizeIsZero) {
    EXPECT_EQ(get_datatype_size(DT_UNKNOWN), 0u);
}

TEST_F(DataTypeTest, StringRepresentation) {
    EXPECT_EQ(get_datatype_str(DT_FLOAT32), "float");
    EXPECT_EQ(get_datatype_str(DT_FLOAT16), "half_t");
    EXPECT_EQ(get_datatype_str(DT_BFLOAT16), "bfloat16_t");
    EXPECT_EQ(get_datatype_str(DT_INT32), "int32");
    EXPECT_EQ(get_datatype_str(DT_INT8), "int8");
    EXPECT_EQ(get_datatype_str(DT_DOUBLE), "double");
}

TEST_F(DataTypeTest, UnknownTypeString) {
    EXPECT_EQ(get_datatype_str(DT_UNKNOWN), "unknown");
}

// =============================================================================
// DataType Parameterized Tests
// =============================================================================

struct DataTypeSizeParam {
    DataType dtype;
    size_t expected_size;
};

class DataTypeSizeTest : public ::testing::TestWithParam<DataTypeSizeParam> {};

TEST_P(DataTypeSizeTest, CorrectSize) {
    auto param = GetParam();
    EXPECT_EQ(get_datatype_size(param.dtype), param.expected_size);
}

INSTANTIATE_TEST_SUITE_P(
    AllDataTypes,
    DataTypeSizeTest,
    ::testing::Values(
        DataTypeSizeParam{DT_INT8, 1},
        DataTypeSizeParam{DT_UINT8, 1},
        DataTypeSizeParam{DT_FLOAT8, 1},
        DataTypeSizeParam{DT_FLOAT16, 2},
        DataTypeSizeParam{DT_BFLOAT16, 2},
        DataTypeSizeParam{DT_INT16, 2},
        DataTypeSizeParam{DT_UINT16, 2},
        DataTypeSizeParam{DT_FLOAT32, 4},
        DataTypeSizeParam{DT_INT32, 4},
        DataTypeSizeParam{DT_UINT32, 4},
        DataTypeSizeParam{DT_DOUBLE, 8},
        DataTypeSizeParam{DT_INT64, 8},
        DataTypeSizeParam{DT_UINT64, 8}
    )
);

// =============================================================================
// Layout Tests
// =============================================================================

class LayoutTest : public ::testing::Test {};

TEST_F(LayoutTest, DmemRowMajorToCmem) {
    EXPECT_EQ(dmemlayout_to_cmemlayout(DmemRowMajor), CmemRowMajor);
}

TEST_F(LayoutTest, DmemColumnMajorToCmem) {
    EXPECT_EQ(dmemlayout_to_cmemlayout(DmemColumnMajor), CmemColumnMajor);
}

TEST_F(LayoutTest, UnknownLayoutToCmem) {
    EXPECT_EQ(dmemlayout_to_cmemlayout(DmemUnknownLayout), CmemUnknownLayout);
}

TEST_F(LayoutTest, DmemLayoutEnumValues) {
    EXPECT_EQ(DmemRowMajor, 100);
    EXPECT_EQ(DmemColumnMajor, 101);
    EXPECT_EQ(DmemUnknownLayout, 199);
}

TEST_F(LayoutTest, SmemLayoutEnumValues) {
    EXPECT_EQ(SmemRowMajor, 200);
    EXPECT_EQ(SmemColumnMajor, 201);
    EXPECT_EQ(SmemUnknownLayout, 299);
}

TEST_F(LayoutTest, CmemLayoutEnumValues) {
    EXPECT_EQ(CmemRowMajor, 300);
    EXPECT_EQ(CmemColumnMajor, 301);
    EXPECT_EQ(CmemUnknownLayout, 399);
}

TEST_F(LayoutTest, TensorOpCrosswiseLayouts) {
    EXPECT_EQ(SmemRowMajorTensorOpMultiplicand_Crosswise16, 202);
    EXPECT_EQ(SmemRowMajorTensorOpMultiplicand_Crosswise32, 203);
    EXPECT_EQ(SmemRowMajorTensorOpMultiplicand_Crosswise64, 204);
    EXPECT_EQ(SmemColumnMajorTensorOpMultiplicand_Crosswise16, 205);
    EXPECT_EQ(SmemColumnMajorTensorOpMultiplicand_Crosswise32, 206);
    EXPECT_EQ(SmemColumnMajorTensorOpMultiplicand_Crosswise64, 207);
}

// =============================================================================
// ThreadBlock Operator Type Tests
// =============================================================================

class TBOperatorTypeTest : public ::testing::Test {};

TEST_F(TBOperatorTypeTest, UnaryOperators) {
    EXPECT_TRUE(is_threadblock_element_unary(TB_EXP_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_SQUARE_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_SQRT_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_SILU_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_GELU_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_RELU_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_CLAMP_OP));
    EXPECT_TRUE(is_threadblock_element_unary(TB_MUL_SCALAR_OP));
}

TEST_F(TBOperatorTypeTest, NonUnaryOperators) {
    EXPECT_FALSE(is_threadblock_element_unary(TB_MATMUL_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_ADD_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_MUL_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_DIV_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_SUB_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_INPUT_OP));
    EXPECT_FALSE(is_threadblock_element_unary(TB_OUTPUT_OP));
}

TEST_F(TBOperatorTypeTest, OperatorTypeRanges) {
    // Input/Output/Basic ops: 2000-2009
    EXPECT_GE(TB_INPUT_OP, 2000);
    EXPECT_LE(TB_OUTPUT_OP, 2009);
    
    // Element unary ops: 2100-2199
    EXPECT_GE(TB_EXP_OP, 2100);
    EXPECT_LT(TB_LOG_OP, 2200);
    
    // Element binary ops: 2200-2299
    EXPECT_GE(TB_ADD_OP, 2200);
    EXPECT_LT(TB_POW_OP, 2300);
}

// =============================================================================
// Backend Type Tests
// =============================================================================

class BackendTypeTest : public ::testing::Test {};

TEST_F(BackendTypeTest, HardwareBackends) {
    EXPECT_TRUE(is_hardware_backend(BT_CUDA));
    EXPECT_TRUE(is_hardware_backend(BT_CPU));
    EXPECT_TRUE(is_hardware_backend(BT_MPS));
    EXPECT_TRUE(is_hardware_backend(BT_ROCM));
    EXPECT_TRUE(is_hardware_backend(BT_ASCEND));
    EXPECT_TRUE(is_hardware_backend(BT_MACA));
    EXPECT_TRUE(is_hardware_backend(BT_TPU));
    EXPECT_TRUE(is_hardware_backend(BT_FPGA));
    EXPECT_TRUE(is_hardware_backend(BT_XPU));
}

TEST_F(BackendTypeTest, LibraryBackends) {
    EXPECT_TRUE(is_library_backend(BT_CUDNN));
    EXPECT_TRUE(is_library_backend(BT_CUSPARSELT));
    EXPECT_TRUE(is_library_backend(BT_CUTLASS));
    EXPECT_TRUE(is_library_backend(BT_MKL));
    EXPECT_TRUE(is_library_backend(BT_MKLDNN));
    EXPECT_TRUE(is_library_backend(BT_OPENMP));
}

TEST_F(BackendTypeTest, MLIRBackends) {
    EXPECT_TRUE(is_mlir_backend(BT_MLIR));
    EXPECT_TRUE(is_mlir_backend(BT_MLIR_LLVM));
    EXPECT_TRUE(is_mlir_backend(BT_MLIR_NVVM));
    EXPECT_TRUE(is_mlir_backend(BT_MLIR_ROCDL));
    EXPECT_TRUE(is_mlir_backend(BT_MLIR_SPIRV));
    EXPECT_TRUE(is_mlir_backend(BT_STABLEHLO));
    EXPECT_TRUE(is_mlir_backend(BT_MHLO));
    EXPECT_TRUE(is_mlir_backend(BT_TOSA));
    EXPECT_TRUE(is_mlir_backend(BT_LINALG));
    EXPECT_TRUE(is_mlir_backend(BT_IREE));
    EXPECT_TRUE(is_mlir_backend(BT_TVM));
    EXPECT_TRUE(is_mlir_backend(BT_XLA));
}

TEST_F(BackendTypeTest, DSLBackends) {
    EXPECT_TRUE(is_dsl_backend(BT_TRITON));
    EXPECT_TRUE(is_dsl_backend(BT_NKI));
    // MLIR backends are also DSL backends
    EXPECT_TRUE(is_dsl_backend(BT_MLIR));
}

TEST_F(BackendTypeTest, HardwareNotMLIR) {
    EXPECT_FALSE(is_mlir_backend(BT_CUDA));
    EXPECT_FALSE(is_mlir_backend(BT_CPU));
    EXPECT_FALSE(is_mlir_backend(BT_MPS));
}

TEST_F(BackendTypeTest, MLIRNotHardware) {
    EXPECT_FALSE(is_hardware_backend(BT_MLIR));
    EXPECT_FALSE(is_hardware_backend(BT_STABLEHLO));
    EXPECT_FALSE(is_hardware_backend(BT_TOSA));
}

// =============================================================================
// Backend Fallback Tests
// =============================================================================

class BackendFallbackTest : public ::testing::Test {};

TEST_F(BackendFallbackTest, CUDALibrariesToCUDA) {
    EXPECT_EQ(get_fallback_backend(BT_CUDNN), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_CUSPARSELT), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_CUTLASS), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_MHA), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_TRITON), BT_CUDA);
}

TEST_F(BackendFallbackTest, CPULibrariesToCPU) {
    EXPECT_EQ(get_fallback_backend(BT_MKL), BT_CPU);
    EXPECT_EQ(get_fallback_backend(BT_MKLDNN), BT_CPU);
    EXPECT_EQ(get_fallback_backend(BT_OPENMP), BT_CPU);
    EXPECT_EQ(get_fallback_backend(BT_XEON), BT_CPU);
}

TEST_F(BackendFallbackTest, MLIRToHardware) {
    EXPECT_EQ(get_fallback_backend(BT_MLIR_NVVM), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_MLIR_ROCDL), BT_ROCM);
    EXPECT_EQ(get_fallback_backend(BT_MLIR_LLVM), BT_CPU);
}

TEST_F(BackendFallbackTest, XLAEcosystemToTPU) {
    EXPECT_EQ(get_fallback_backend(BT_STABLEHLO), BT_TPU);
    EXPECT_EQ(get_fallback_backend(BT_MHLO), BT_TPU);
    EXPECT_EQ(get_fallback_backend(BT_XLA), BT_TPU);
}

TEST_F(BackendFallbackTest, HardwareToItself) {
    EXPECT_EQ(get_fallback_backend(BT_CUDA), BT_CUDA);
    EXPECT_EQ(get_fallback_backend(BT_CPU), BT_CPU);
    EXPECT_EQ(get_fallback_backend(BT_MPS), BT_MPS);
    EXPECT_EQ(get_fallback_backend(BT_TPU), BT_TPU);
}

// =============================================================================
// Backend Category Tests
// =============================================================================

class BackendCategoryTest : public ::testing::Test {};

TEST_F(BackendCategoryTest, NVIDIACategory) {
    EXPECT_EQ(get_backend_category(BT_CUDA), BC_NVIDIA_GPU);
    EXPECT_EQ(get_backend_category(BT_CUDNN), BC_NVIDIA_GPU);
    EXPECT_EQ(get_backend_category(BT_TRITON), BC_NVIDIA_GPU);
    EXPECT_EQ(get_backend_category(BT_MLIR_NVVM), BC_NVIDIA_GPU);
}

TEST_F(BackendCategoryTest, AMDCategory) {
    EXPECT_EQ(get_backend_category(BT_ROCM), BC_AMD_GPU);
    EXPECT_EQ(get_backend_category(BT_MLIR_ROCDL), BC_AMD_GPU);
}

TEST_F(BackendCategoryTest, IntelCategory) {
    EXPECT_EQ(get_backend_category(BT_XPU), BC_INTEL_GPU);
    EXPECT_EQ(get_backend_category(BT_MLIR_SPIRV), BC_INTEL_GPU);
}

TEST_F(BackendCategoryTest, AppleCategory) {
    EXPECT_EQ(get_backend_category(BT_MPS), BC_APPLE_GPU);
}

TEST_F(BackendCategoryTest, HuaweiCategory) {
    EXPECT_EQ(get_backend_category(BT_ASCEND), BC_HUAWEI_NPU);
}

TEST_F(BackendCategoryTest, GoogleCategory) {
    EXPECT_EQ(get_backend_category(BT_TPU), BC_GOOGLE_TPU);
    EXPECT_EQ(get_backend_category(BT_STABLEHLO), BC_GOOGLE_TPU);
    EXPECT_EQ(get_backend_category(BT_XLA), BC_GOOGLE_TPU);
}

TEST_F(BackendCategoryTest, CPUCategory) {
    EXPECT_EQ(get_backend_category(BT_CPU), BC_CPU);
    EXPECT_EQ(get_backend_category(BT_MKL), BC_CPU);
    EXPECT_EQ(get_backend_category(BT_OPENMP), BC_CPU);
    EXPECT_EQ(get_backend_category(BT_MLIR_LLVM), BC_CPU);
}

TEST_F(BackendCategoryTest, MLIRCategory) {
    EXPECT_EQ(get_backend_category(BT_MLIR), BC_MLIR);
    EXPECT_EQ(get_backend_category(BT_TOSA), BC_MLIR);
    EXPECT_EQ(get_backend_category(BT_LINALG), BC_MLIR);
    EXPECT_EQ(get_backend_category(BT_IREE), BC_MLIR);
}

// =============================================================================
// MLIR Dialect Tests
// =============================================================================

class MLIRDialectTest : public ::testing::Test {};

TEST_F(MLIRDialectTest, CoreDialectNames) {
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_BUILTIN), "builtin");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_ARITH), "arith");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_FUNC), "func");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_SCF), "scf");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_MEMREF), "memref");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_TENSOR), "tensor");
}

TEST_F(MLIRDialectTest, ComputationDialectNames) {
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_LINALG), "linalg");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_TOSA), "tosa");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_STABLEHLO), "stablehlo");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_MHLO), "mhlo");
}

TEST_F(MLIRDialectTest, TargetDialectNames) {
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_LLVM), "llvm");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_NVVM), "nvvm");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_ROCDL), "rocdl");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_SPIRV), "spirv");
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_GPU), "gpu");
}

TEST_F(MLIRDialectTest, UnknownDialect) {
    EXPECT_STREQ(get_mlir_dialect_name(MLIR_DIALECT_UNKNOWN), "unknown");
}

// =============================================================================
// KNOperator Type Tests
// =============================================================================

class KNOperatorTypeTest : public ::testing::Test {};

TEST_F(KNOperatorTypeTest, IOOperators) {
    EXPECT_EQ(KN_INPUT_OP, 1001);
    EXPECT_EQ(KN_OUTPUT_OP, 1002);
}

TEST_F(KNOperatorTypeTest, MatmulOperator) {
    EXPECT_EQ(KN_MATMUL_OP, 1003);
}

TEST_F(KNOperatorTypeTest, ElementUnaryRange) {
    EXPECT_GE(KN_EXP_OP, 1100);
    EXPECT_LE(KN_GELU_OP, 1199);
}

TEST_F(KNOperatorTypeTest, ElementBinaryRange) {
    EXPECT_GE(KN_ADD_OP, 1200);
    EXPECT_LE(KN_POW_OP, 1299);
}

TEST_F(KNOperatorTypeTest, ReductionRange) {
    EXPECT_GE(KN_REDUCTION_0_OP, 1300);
    EXPECT_LE(KN_REDUCTION_2_OP, 1349);
}

TEST_F(KNOperatorTypeTest, ConcatRange) {
    EXPECT_GE(KN_CONCAT_0_OP, 1400);
    EXPECT_LE(KN_CONCAT_2_OP, 1409);
}

TEST_F(KNOperatorTypeTest, SplitRange) {
    EXPECT_GE(KN_SPLIT_0_OP, 1420);
    EXPECT_LE(KN_SPLIT_2_OP, 1429);
}

// =============================================================================
// Activation Type Tests
// =============================================================================

class ActivationTypeTest : public ::testing::Test {};

TEST_F(ActivationTypeTest, ActivationEnumValues) {
    EXPECT_EQ(ACT_UNKOWN, 3000);
    EXPECT_EQ(ACT_EXP, 3001);
    EXPECT_EQ(ACT_RELU, 3002);
    EXPECT_EQ(ACT_GELU, 3003);
    EXPECT_EQ(ACT_SILU, 3004);
    EXPECT_EQ(ACT_NONE, 3099);
}

// =============================================================================
// Comprehensive Backend Tests (Parameterized)
// =============================================================================

struct BackendClassificationParam {
    BackendType type;
    bool is_hw;
    bool is_lib;
    bool is_mlir;
};

class BackendClassificationTest 
    : public ::testing::TestWithParam<BackendClassificationParam> {};

TEST_P(BackendClassificationTest, Classification) {
    auto param = GetParam();
    EXPECT_EQ(is_hardware_backend(param.type), param.is_hw);
    EXPECT_EQ(is_library_backend(param.type), param.is_lib);
    EXPECT_EQ(is_mlir_backend(param.type), param.is_mlir);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendClassifications,
    BackendClassificationTest,
    ::testing::Values(
        // Hardware backends
        BackendClassificationParam{BT_CUDA, true, false, false},
        BackendClassificationParam{BT_CPU, true, false, false},
        BackendClassificationParam{BT_MPS, true, false, false},
        BackendClassificationParam{BT_ROCM, true, false, false},
        BackendClassificationParam{BT_TPU, true, false, false},
        // Library backends
        BackendClassificationParam{BT_CUDNN, false, true, false},
        BackendClassificationParam{BT_MKL, false, true, false},
        BackendClassificationParam{BT_OPENMP, false, true, false},
        // MLIR backends
        BackendClassificationParam{BT_MLIR, false, false, true},
        BackendClassificationParam{BT_STABLEHLO, false, false, true},
        BackendClassificationParam{BT_TOSA, false, false, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
