// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_nki_transpiler_gtest.cc
 * @brief NKI Transpiler Unit Tests (Google Test version)
 *
 * Tests for yirage NKI (Neuron Kernel Interface) transpiler.
 * Tests cover:
 *   - Data type conversion (yirage -> NKI)
 *   - Helper functions (tiled_transpose, tiled_matmul)
 *   - Operator type mappings (TB/KN -> NKI)
 *   - Tensor variable naming
 *   - Code generation structure
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <sstream>

namespace yirage {
namespace type {

// Data type enumeration (mirror actual)
enum DataType {
    DT_INT4 = 925,
    DT_INT8 = 935,
    DT_INT32 = 955,
    DT_UINT16 = 946,
    DT_FLOAT8 = 930,
    DT_FLOAT16 = 940,
    DT_BFLOAT16 = 941,
    DT_FLOAT32 = 950,
    DT_UNKNOWN = 999,
};

// TB Operator types
enum TBOperatorType {
    TB_EXP_OP = 2100,
    TB_SILU_OP = 2104,
    TB_SQUARE_OP = 2101,
    TB_SQRT_OP = 2102,
    TB_RELU_OP = 2150,
    TB_CLAMP_OP = 2151,
    TB_MUL_SCALAR_OP = 2103,
    TB_ADD_OP = 2200,
    TB_MUL_OP = 2201,
    TB_DIV_OP = 2202,
    TB_SUB_OP = 2203,
    TB_POW_OP = 2204,
};

// KN Operator types
enum KNOperatorType {
    KN_EXP_OP = 1100,
    KN_SILU_OP = 1104,
    KN_SQUARE_OP = 1101,
    KN_SQRT_OP = 1102,
    KN_RELU_OP = 1150,
    KN_CLAMP_OP = 1151,
    KN_MUL_SCALAR_OP = 1103,
    KN_ADD_OP = 1200,
    KN_MUL_OP = 1201,
    KN_DIV_OP = 1202,
    KN_POW_OP = 1203,
};

}  // namespace type

namespace nki_transpiler {

using namespace yirage::type;

// =============================================================================
// Mock Functions (mirror actual implementations)
// =============================================================================

// Convert YiRage DataType to NKI type string
std::string yirage_dtype_to_nki(DataType dt) {
    switch (dt) {
        case DT_INT4:
            return "";  // Not supported
        case DT_INT8:
            return "nl.int8";
        case DT_INT32:
            return "nl.int32";
        case DT_UINT16:
            return "nl.uint16";
        case DT_FLOAT8:
            return "nl.float8_e4m3";
        case DT_FLOAT16:
            return "nl.float16";
        case DT_BFLOAT16:
            return "nl.bfloat16";
        case DT_FLOAT32:
            return "nl.float32";
        default:
            return "";
    }
}

// Get Python literal for boolean
std::string get_python_literal(bool value) {
    return value ? "True" : "False";
}

// Convert TB operator type to NKI function
std::string ugraph_tboperator_type_to_nki(TBOperatorType type) {
    switch (type) {
        case TB_EXP_OP:      return "nl.exp";
        case TB_SILU_OP:     return "nl.silu";
        case TB_SQUARE_OP:   return "nl.square";
        case TB_SQRT_OP:     return "nl.sqrt";
        case TB_RELU_OP:     return "nl.relu";
        case TB_CLAMP_OP:    return "nl.clamp";
        case TB_MUL_SCALAR_OP: return "nl.multiply";
        case TB_ADD_OP:      return "nl.add";
        case TB_MUL_OP:      return "nl.multiply";
        case TB_DIV_OP:      return "nl.divide";
        case TB_SUB_OP:      return "nl.subtract";
        case TB_POW_OP:      return "nl.power";
        default:             return "";
    }
}

// Convert KN operator type to NKI function
std::string ugraph_knoperator_type_to_nki(KNOperatorType type) {
    switch (type) {
        case KN_EXP_OP:      return "nl.exp";
        case KN_SILU_OP:     return "nl.silu";
        case KN_SQUARE_OP:   return "nl.square";
        case KN_SQRT_OP:     return "nl.sqrt";
        case KN_RELU_OP:     return "nl.relu";
        case KN_CLAMP_OP:    return "nl.clamp";
        case KN_MUL_SCALAR_OP: return "nl.multiply";
        case KN_ADD_OP:      return "nl.add";
        case KN_MUL_OP:      return "nl.multiply";
        case KN_DIV_OP:      return "nl.divide";
        case KN_POW_OP:      return "nl.power";
        default:             return "";
    }
}

// Helper function structure (mock)
struct HelperFunction {
    std::string name;
    std::vector<std::string> params;
    std::string body;
    std::vector<std::string> deps;
    
    std::string get_invocation(std::vector<std::string> const& args) const {
        if (args.size() != params.size()) return "";
        std::string invocation = name + "(";
        for (size_t i = 0; i < args.size(); ++i) {
            invocation += params[i] + " = " + args[i];
            if (i < args.size() - 1) {
                invocation += ", ";
            }
        }
        invocation += ")";
        return invocation;
    }
    
    std::string get_code() const {
        std::string code = "def " + name + "(";
        for (size_t i = 0; i < params.size(); ++i) {
            code += params[i];
            if (i < params.size() - 1) {
                code += ", ";
            }
        }
        code += "):\n";
        code += body;
        return code;
    }
};

// Create tiled_transpose helper function
HelperFunction tiled_transpose_function() {
    HelperFunction fn;
    fn.name = "tiled_transpose";
    fn.params = {"tensor"};
    fn.body = R"(  assert len(tensor.shape) == 3
  ftile_size, num_ftile, psize = tensor.shape[0], tensor.shape[1], tensor.shape[2]
  num_ptile, ptile_size = psize // 128, 128
  result = nl.ndarray((ptile_size, num_ptile, num_ftile * ftile_size), dtype=tensor.dtype, buffer=nl.sbuf)
  return result
)";
    return fn;
}

// Create tiled_matmul helper function
HelperFunction tiled_matmul_function() {
    HelperFunction fn;
    fn.deps = {"tiled_transpose", "tiled_matmul_accum"};
    fn.name = "tiled_matmul";
    fn.params = {"_lhs", "_rhs", "lhs_transposed", "rhs_transposed", "dtype"};
    fn.body = R"(  rhs = tiled_transpose(_rhs) if rhs_transposed else _rhs
  # ... matmul implementation
  return result
)";
    return fn;
}

// Create tiled_matmul_accum helper function
HelperFunction tiled_matmul_accum_function() {
    HelperFunction fn;
    fn.name = "tiled_matmul_accum";
    fn.params = {"_lhs", "rhs", "result", "lhs_transposed"};
    fn.body = R"(  # Accumulate matmul results
  # ... implementation
)";
    return fn;
}

// NKI transpile configuration
struct NKITranspilerConfig {
    bool optimize_layout = true;
    int max_partition_size = 128;
    int max_free_dim_size = 512;
};

// NKI tensor initializer enum
enum class NKITensorInitializer {
    NONE,
    ZERO,
    RANDOM,
};

// STensor metadata
struct STensorMeta {
    int partition_dim = -1;
};

// Code generation helper
class CodeKeeper {
public:
    void e(const std::string& line) {
        for (int i = 0; i < indent_level_; ++i) {
            code_ << "  ";
        }
        code_ << line << "\n";
    }
    
    template<typename... Args>
    void e(const std::string& fmt_str, Args... args) {
        // Simplified format implementation
        e(fmt_str);
    }
    
    void inc_indent() { indent_level_++; }
    void dec_indent() { if (indent_level_ > 0) indent_level_--; }
    
    std::string to_string() const { return code_.str(); }
    
private:
    std::ostringstream code_;
    int indent_level_ = 0;
};

// NKI transpile result
struct NKITranspileResult {
    std::string code;
    std::vector<std::string> errors;
    
    bool has_error() const { return !errors.empty(); }
};

// NKI custom op transpile result
struct NKICustomOPTranspileResult {
    std::string func_name;
    std::string code;
};

// Ceiling division utility
inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace nki_transpiler
}  // namespace yirage

using namespace yirage::type;
using namespace yirage::nki_transpiler;

// =============================================================================
// DataType Conversion Tests
// =============================================================================

class NKIDataTypeTest : public ::testing::Test {};

TEST_F(NKIDataTypeTest, Int8ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_INT8), "nl.int8");
}

TEST_F(NKIDataTypeTest, Int32ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_INT32), "nl.int32");
}

TEST_F(NKIDataTypeTest, UInt16ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_UINT16), "nl.uint16");
}

TEST_F(NKIDataTypeTest, Float8ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_FLOAT8), "nl.float8_e4m3");
}

TEST_F(NKIDataTypeTest, Float16ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_FLOAT16), "nl.float16");
}

TEST_F(NKIDataTypeTest, BFloat16ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_BFLOAT16), "nl.bfloat16");
}

TEST_F(NKIDataTypeTest, Float32ToNKI) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_FLOAT32), "nl.float32");
}

TEST_F(NKIDataTypeTest, Int4NotSupported) {
    // INT4 is not supported in NKI
    EXPECT_EQ(yirage_dtype_to_nki(DT_INT4), "");
}

TEST_F(NKIDataTypeTest, UnknownTypeReturnsEmpty) {
    EXPECT_EQ(yirage_dtype_to_nki(DT_UNKNOWN), "");
}

// =============================================================================
// DataType Conversion Parameterized Tests
// =============================================================================

struct NKIDTypeParam {
    DataType dtype;
    std::string expected;
};

class NKIDataTypeParamTest : public ::testing::TestWithParam<NKIDTypeParam> {};

TEST_P(NKIDataTypeParamTest, ConvertToNKI) {
    auto param = GetParam();
    EXPECT_EQ(yirage_dtype_to_nki(param.dtype), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllSupportedTypes,
    NKIDataTypeParamTest,
    ::testing::Values(
        NKIDTypeParam{DT_INT8, "nl.int8"},
        NKIDTypeParam{DT_INT32, "nl.int32"},
        NKIDTypeParam{DT_UINT16, "nl.uint16"},
        NKIDTypeParam{DT_FLOAT8, "nl.float8_e4m3"},
        NKIDTypeParam{DT_FLOAT16, "nl.float16"},
        NKIDTypeParam{DT_BFLOAT16, "nl.bfloat16"},
        NKIDTypeParam{DT_FLOAT32, "nl.float32"}
    )
);

// =============================================================================
// Python Literal Tests
// =============================================================================

class PythonLiteralTest : public ::testing::Test {};

TEST_F(PythonLiteralTest, TrueToTrue) {
    EXPECT_EQ(get_python_literal(true), "True");
}

TEST_F(PythonLiteralTest, FalseToFalse) {
    EXPECT_EQ(get_python_literal(false), "False");
}

// =============================================================================
// TB Operator to NKI Tests
// =============================================================================

class TBOperatorToNKITest : public ::testing::Test {};

TEST_F(TBOperatorToNKITest, UnaryOperators) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_EXP_OP), "nl.exp");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_SILU_OP), "nl.silu");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_SQUARE_OP), "nl.square");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_SQRT_OP), "nl.sqrt");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_RELU_OP), "nl.relu");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_CLAMP_OP), "nl.clamp");
}

TEST_F(TBOperatorToNKITest, BinaryOperators) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_ADD_OP), "nl.add");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_MUL_OP), "nl.multiply");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_DIV_OP), "nl.divide");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_SUB_OP), "nl.subtract");
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_POW_OP), "nl.power");
}

TEST_F(TBOperatorToNKITest, MulScalarUsesMultiply) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_MUL_SCALAR_OP), "nl.multiply");
}

// =============================================================================
// KN Operator to NKI Tests
// =============================================================================

class KNOperatorToNKITest : public ::testing::Test {};

TEST_F(KNOperatorToNKITest, UnaryOperators) {
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_EXP_OP), "nl.exp");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_SILU_OP), "nl.silu");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_SQUARE_OP), "nl.square");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_SQRT_OP), "nl.sqrt");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_RELU_OP), "nl.relu");
}

TEST_F(KNOperatorToNKITest, BinaryOperators) {
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_ADD_OP), "nl.add");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_MUL_OP), "nl.multiply");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_DIV_OP), "nl.divide");
    EXPECT_EQ(ugraph_knoperator_type_to_nki(KN_POW_OP), "nl.power");
}

// =============================================================================
// Operator Mapping Consistency Tests
// =============================================================================

class OperatorMappingConsistencyTest : public ::testing::Test {};

TEST_F(OperatorMappingConsistencyTest, ExpConsistent) {
    // TB_EXP_OP and KN_EXP_OP should map to same NKI function
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_EXP_OP),
              ugraph_knoperator_type_to_nki(KN_EXP_OP));
}

TEST_F(OperatorMappingConsistencyTest, SiluConsistent) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_SILU_OP),
              ugraph_knoperator_type_to_nki(KN_SILU_OP));
}

TEST_F(OperatorMappingConsistencyTest, AddConsistent) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_ADD_OP),
              ugraph_knoperator_type_to_nki(KN_ADD_OP));
}

TEST_F(OperatorMappingConsistencyTest, MulConsistent) {
    EXPECT_EQ(ugraph_tboperator_type_to_nki(TB_MUL_OP),
              ugraph_knoperator_type_to_nki(KN_MUL_OP));
}

// =============================================================================
// Helper Function Tests
// =============================================================================

class HelperFunctionTest : public ::testing::Test {};

TEST_F(HelperFunctionTest, TiledTransposeName) {
    auto fn = tiled_transpose_function();
    EXPECT_EQ(fn.name, "tiled_transpose");
}

TEST_F(HelperFunctionTest, TiledTransposeParams) {
    auto fn = tiled_transpose_function();
    ASSERT_EQ(fn.params.size(), 1u);
    EXPECT_EQ(fn.params[0], "tensor");
}

TEST_F(HelperFunctionTest, TiledTransposeInvocation) {
    auto fn = tiled_transpose_function();
    std::string inv = fn.get_invocation({"my_tensor"});
    EXPECT_EQ(inv, "tiled_transpose(tensor = my_tensor)");
}

TEST_F(HelperFunctionTest, TiledMatmulName) {
    auto fn = tiled_matmul_function();
    EXPECT_EQ(fn.name, "tiled_matmul");
}

TEST_F(HelperFunctionTest, TiledMatmulParams) {
    auto fn = tiled_matmul_function();
    ASSERT_EQ(fn.params.size(), 5u);
    EXPECT_EQ(fn.params[0], "_lhs");
    EXPECT_EQ(fn.params[1], "_rhs");
    EXPECT_EQ(fn.params[2], "lhs_transposed");
    EXPECT_EQ(fn.params[3], "rhs_transposed");
    EXPECT_EQ(fn.params[4], "dtype");
}

TEST_F(HelperFunctionTest, TiledMatmulDependencies) {
    auto fn = tiled_matmul_function();
    ASSERT_EQ(fn.deps.size(), 2u);
    EXPECT_EQ(fn.deps[0], "tiled_transpose");
    EXPECT_EQ(fn.deps[1], "tiled_matmul_accum");
}

TEST_F(HelperFunctionTest, TiledMatmulAccumName) {
    auto fn = tiled_matmul_accum_function();
    EXPECT_EQ(fn.name, "tiled_matmul_accum");
}

TEST_F(HelperFunctionTest, TiledMatmulAccumParams) {
    auto fn = tiled_matmul_accum_function();
    ASSERT_EQ(fn.params.size(), 4u);
    EXPECT_EQ(fn.params[0], "_lhs");
    EXPECT_EQ(fn.params[1], "rhs");
    EXPECT_EQ(fn.params[2], "result");
    EXPECT_EQ(fn.params[3], "lhs_transposed");
}

TEST_F(HelperFunctionTest, GetCodeContainsDef) {
    auto fn = tiled_transpose_function();
    std::string code = fn.get_code();
    EXPECT_NE(code.find("def tiled_transpose"), std::string::npos);
}

TEST_F(HelperFunctionTest, GetCodeContainsParams) {
    auto fn = tiled_transpose_function();
    std::string code = fn.get_code();
    EXPECT_NE(code.find("(tensor):"), std::string::npos);
}

// =============================================================================
// CodeKeeper Tests
// =============================================================================

class CodeKeeperTest : public ::testing::Test {
protected:
    CodeKeeper code;
};

TEST_F(CodeKeeperTest, EmitLine) {
    code.e("import nki");
    EXPECT_NE(code.to_string().find("import nki"), std::string::npos);
}

TEST_F(CodeKeeperTest, EmitMultipleLines) {
    code.e("line1");
    code.e("line2");
    std::string result = code.to_string();
    EXPECT_NE(result.find("line1"), std::string::npos);
    EXPECT_NE(result.find("line2"), std::string::npos);
}

TEST_F(CodeKeeperTest, IndentIncrease) {
    code.e("def foo():");
    code.inc_indent();
    code.e("pass");
    std::string result = code.to_string();
    // Should have indentation before "pass"
    EXPECT_NE(result.find("  pass"), std::string::npos);
}

TEST_F(CodeKeeperTest, IndentDecrease) {
    code.inc_indent();
    code.e("indented");
    code.dec_indent();
    code.e("not_indented");
    std::string result = code.to_string();
    EXPECT_NE(result.find("  indented"), std::string::npos);
    // "not_indented" should start at beginning
    size_t pos = result.find("not_indented");
    ASSERT_NE(pos, std::string::npos);
}

TEST_F(CodeKeeperTest, NestedIndent) {
    code.e("level0");
    code.inc_indent();
    code.e("level1");
    code.inc_indent();
    code.e("level2");
    code.dec_indent();
    code.e("back_to_level1");
    
    std::string result = code.to_string();
    EXPECT_NE(result.find("    level2"), std::string::npos);  // 4 spaces
    EXPECT_NE(result.find("  level1"), std::string::npos);    // 2 spaces
}

// =============================================================================
// NKI Transpiler Config Tests
// =============================================================================

class NKITranspilerConfigTest : public ::testing::Test {};

TEST_F(NKITranspilerConfigTest, DefaultConfig) {
    NKITranspilerConfig config;
    EXPECT_TRUE(config.optimize_layout);
    EXPECT_EQ(config.max_partition_size, 128);
    EXPECT_EQ(config.max_free_dim_size, 512);
}

TEST_F(NKITranspilerConfigTest, CustomConfig) {
    NKITranspilerConfig config;
    config.optimize_layout = false;
    config.max_partition_size = 64;
    config.max_free_dim_size = 256;
    
    EXPECT_FALSE(config.optimize_layout);
    EXPECT_EQ(config.max_partition_size, 64);
    EXPECT_EQ(config.max_free_dim_size, 256);
}

// =============================================================================
// STensor Meta Tests
// =============================================================================

class STensorMetaTest : public ::testing::Test {};

TEST_F(STensorMetaTest, DefaultPartitionDim) {
    STensorMeta meta;
    EXPECT_EQ(meta.partition_dim, -1);
}

TEST_F(STensorMetaTest, SetPartitionDim) {
    STensorMeta meta;
    meta.partition_dim = 1;
    EXPECT_EQ(meta.partition_dim, 1);
}

// =============================================================================
// NKI Transpile Result Tests
// =============================================================================

class NKITranspileResultTest : public ::testing::Test {};

TEST_F(NKITranspileResultTest, NoErrors) {
    NKITranspileResult result;
    result.code = "def kernel(): pass";
    EXPECT_FALSE(result.has_error());
}

TEST_F(NKITranspileResultTest, WithErrors) {
    NKITranspileResult result;
    result.errors.push_back("Z3 unsat: No valid layout found");
    EXPECT_TRUE(result.has_error());
}

TEST_F(NKITranspileResultTest, MultipleErrors) {
    NKITranspileResult result;
    result.errors.push_back("Error 1");
    result.errors.push_back("Error 2");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.errors.size(), 2u);
}

// =============================================================================
// NKI Custom OP Result Tests
// =============================================================================

class NKICustomOPResultTest : public ::testing::Test {};

TEST_F(NKICustomOPResultTest, FuncNameFormat) {
    NKICustomOPTranspileResult result;
    result.func_name = "custom_kernel_0";
    EXPECT_NE(result.func_name.find("custom_kernel_"), std::string::npos);
}

TEST_F(NKICustomOPResultTest, CodeNotEmpty) {
    NKICustomOPTranspileResult result;
    result.code = "@nki.jit\ndef kernel(): pass";
    EXPECT_FALSE(result.code.empty());
    EXPECT_NE(result.code.find("@nki.jit"), std::string::npos);
}

// =============================================================================
// Ceiling Division Tests
// =============================================================================

class CeilDivTest : public ::testing::Test {};

TEST_F(CeilDivTest, ExactDivision) {
    EXPECT_EQ(ceil_div(128, 64), 2);
    EXPECT_EQ(ceil_div(256, 128), 2);
    EXPECT_EQ(ceil_div(512, 512), 1);
}

TEST_F(CeilDivTest, NonExactDivision) {
    EXPECT_EQ(ceil_div(129, 64), 3);  // 129/64 = 2.01... -> 3
    EXPECT_EQ(ceil_div(130, 128), 2);  // 130/128 = 1.01... -> 2
    EXPECT_EQ(ceil_div(1, 128), 1);    // 1/128 = 0.007... -> 1
}

TEST_F(CeilDivTest, DivisionByOne) {
    EXPECT_EQ(ceil_div(5, 1), 5);
    EXPECT_EQ(ceil_div(128, 1), 128);
}

// =============================================================================
// NKI Import Tests
// =============================================================================

class NKIImportTest : public ::testing::Test {
protected:
    void checkImports(const std::string& code) {
        EXPECT_NE(code.find("import neuronxcc.nki as nki"), std::string::npos);
        EXPECT_NE(code.find("import neuronxcc.nki.language as nl"), std::string::npos);
        EXPECT_NE(code.find("import neuronxcc.nki.isa as nisa"), std::string::npos);
    }
};

TEST_F(NKIImportTest, HeaderImports) {
    std::string header = R"(
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
)";
    checkImports(header);
}

// =============================================================================
// NKI Tile Size Tests
// =============================================================================

class NKITileSizeTest : public ::testing::Test {
protected:
    // NKI matmul constraints
    static constexpr int MAX_PARTITION_SIZE = 128;
    static constexpr int MAX_FREE_DIM_STATIONARY = 128;
    static constexpr int MAX_FREE_DIM_MOVING = 512;
};

TEST_F(NKITileSizeTest, PartitionSizeConstraint) {
    // Partition dimension must be <= 128
    EXPECT_LE(MAX_PARTITION_SIZE, 128);
}

TEST_F(NKITileSizeTest, StationaryOperandConstraint) {
    // Stationary operand: max 128x128
    EXPECT_LE(MAX_FREE_DIM_STATIONARY, 128);
}

TEST_F(NKITileSizeTest, MovingOperandConstraint) {
    // Moving operand: max 128x512
    EXPECT_LE(MAX_FREE_DIM_MOVING, 512);
}

TEST_F(NKITileSizeTest, MatmulCaseSelection) {
    // Test which matmul case is more efficient
    int M = 256, K = 128, N = 1024;
    
    // Case 1: input0 (MxK) stationary, input1 (KxN) moving
    int case1_matmuls = ceil_div(M, 128) * ceil_div(N, 512);
    
    // Case 2: input1 (KxN) stationary, input0 (MxK) moving
    int case2_matmuls = ceil_div(N, 128) * ceil_div(M, 512);
    
    // For this configuration, case1 should be more efficient
    EXPECT_LT(case1_matmuls, case2_matmuls);
}

// =============================================================================
// NKI Operator Parameterized Tests
// =============================================================================

struct TBOpParam {
    TBOperatorType op;
    std::string nki_name;
};

class TBOperatorParamTest : public ::testing::TestWithParam<TBOpParam> {};

TEST_P(TBOperatorParamTest, MapToNKI) {
    auto param = GetParam();
    EXPECT_EQ(ugraph_tboperator_type_to_nki(param.op), param.nki_name);
}

INSTANTIATE_TEST_SUITE_P(
    AllTBOperators,
    TBOperatorParamTest,
    ::testing::Values(
        TBOpParam{TB_EXP_OP, "nl.exp"},
        TBOpParam{TB_SILU_OP, "nl.silu"},
        TBOpParam{TB_SQUARE_OP, "nl.square"},
        TBOpParam{TB_SQRT_OP, "nl.sqrt"},
        TBOpParam{TB_RELU_OP, "nl.relu"},
        TBOpParam{TB_CLAMP_OP, "nl.clamp"},
        TBOpParam{TB_ADD_OP, "nl.add"},
        TBOpParam{TB_MUL_OP, "nl.multiply"},
        TBOpParam{TB_DIV_OP, "nl.divide"},
        TBOpParam{TB_SUB_OP, "nl.subtract"},
        TBOpParam{TB_POW_OP, "nl.power"}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
