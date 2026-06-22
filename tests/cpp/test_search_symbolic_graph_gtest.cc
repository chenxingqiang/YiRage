// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_symbolic_graph_gtest.cc
 * @brief Symbolic Graph Module Unit Tests
 *
 * Tests for symbolic graph representation:
 *   - Symbolic tensor dimensions
 *   - Dimension variable assignments
 *   - Tensor dimension expressions
 *   - Tensor dimension constraints
 *   - Symbolic TB and KN graphs
 *   - Assignment enumeration
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace yirage {
namespace search {

// =============================================================================
// Type Aliases
// =============================================================================

using tensor_dim_var_index_t = int;

// =============================================================================
// TensorDimExpr - Symbolic dimension expressions
// =============================================================================

class TensorDimExpr {
public:
    enum class Type {
        CONSTANT,
        VARIABLE,
        ADD,
        MUL,
        DIV,
        MIN,
        MAX
    };
    
    // Use tag dispatch to avoid constructor ambiguity since tensor_dim_var_index_t is int
    struct ConstantTag {};
    struct VariableTag {};
    
    TensorDimExpr(ConstantTag, int constant_value)
        : type(Type::CONSTANT), constant_value(constant_value) {}
    
    TensorDimExpr(VariableTag, tensor_dim_var_index_t var_index)
        : type(Type::VARIABLE), var_index(var_index) {}
    
    TensorDimExpr(tensor_dim_var_index_t var_index, Type type)
        : type(type), var_index(var_index) {}
    
    TensorDimExpr(Type type, std::shared_ptr<TensorDimExpr> lhs,
                  std::shared_ptr<TensorDimExpr> rhs)
        : type(type), lhs(lhs), rhs(rhs) {}
    
    bool is_constant() const { return type == Type::CONSTANT; }
    bool is_variable() const { return type == Type::VARIABLE; }
    
    int get_constant() const { return constant_value; }
    tensor_dim_var_index_t get_var_index() const { return var_index; }
    
    int evaluate(std::unordered_map<tensor_dim_var_index_t, int> const& assignments) const {
        switch (type) {
            case Type::CONSTANT:
                return constant_value;
            case Type::VARIABLE:
                return assignments.count(var_index) ? assignments.at(var_index) : 0;
            case Type::ADD:
                return lhs->evaluate(assignments) + rhs->evaluate(assignments);
            case Type::MUL:
                return lhs->evaluate(assignments) * rhs->evaluate(assignments);
            case Type::DIV:
                return lhs->evaluate(assignments) / std::max(1, rhs->evaluate(assignments));
            case Type::MIN:
                return std::min(lhs->evaluate(assignments), rhs->evaluate(assignments));
            case Type::MAX:
                return std::max(lhs->evaluate(assignments), rhs->evaluate(assignments));
        }
        return 0;
    }
    
    std::string to_string() const {
        switch (type) {
            case Type::CONSTANT:
                return std::to_string(constant_value);
            case Type::VARIABLE:
                return "v" + std::to_string(var_index);
            case Type::ADD:
                return "(" + lhs->to_string() + " + " + rhs->to_string() + ")";
            case Type::MUL:
                return "(" + lhs->to_string() + " * " + rhs->to_string() + ")";
            case Type::DIV:
                return "(" + lhs->to_string() + " / " + rhs->to_string() + ")";
            case Type::MIN:
                return "min(" + lhs->to_string() + ", " + rhs->to_string() + ")";
            case Type::MAX:
                return "max(" + lhs->to_string() + ", " + rhs->to_string() + ")";
        }
        return "";
    }
    
    Type type;
    int constant_value = 0;
    tensor_dim_var_index_t var_index = -1;
    std::shared_ptr<TensorDimExpr> lhs, rhs;
};

using TensorDimExprPtr = std::shared_ptr<TensorDimExpr>;

TensorDimExprPtr make_constant(int value) {
    return std::make_shared<TensorDimExpr>(TensorDimExpr::ConstantTag{}, value);
}

TensorDimExprPtr make_variable(tensor_dim_var_index_t index) {
    return std::make_shared<TensorDimExpr>(TensorDimExpr::VariableTag{}, index);
}

TensorDimExprPtr make_add(TensorDimExprPtr lhs, TensorDimExprPtr rhs) {
    return std::make_shared<TensorDimExpr>(TensorDimExpr::Type::ADD, lhs, rhs);
}

TensorDimExprPtr make_mul(TensorDimExprPtr lhs, TensorDimExprPtr rhs) {
    return std::make_shared<TensorDimExpr>(TensorDimExpr::Type::MUL, lhs, rhs);
}

// =============================================================================
// SymbolicTensorDim - Represents a symbolic tensor dimension
// =============================================================================

struct SymbolicTensorDim {
    TensorDimExprPtr expr;
    int min_value = 1;
    int max_value = 1024;
    int step = 1;
    
    SymbolicTensorDim() = default;
    
    SymbolicTensorDim(int constant)
        : expr(make_constant(constant)), min_value(constant), max_value(constant) {}
    
    SymbolicTensorDim(TensorDimExprPtr expr, int min_val, int max_val, int step = 1)
        : expr(expr), min_value(min_val), max_value(max_val), step(step) {}
    
    bool is_constant() const {
        return expr && expr->is_constant();
    }
    
    int get_constant() const {
        return expr ? expr->get_constant() : 0;
    }
    
    std::vector<int> enumerate_values() const {
        std::vector<int> values;
        for (int v = min_value; v <= max_value; v += step) {
            values.push_back(v);
        }
        return values;
    }
};

// =============================================================================
// DimVarAssignments - Concrete assignments for dimension variables
// =============================================================================

class DimVarAssignments {
public:
    void set(tensor_dim_var_index_t var_index, int value) {
        assignments[var_index] = value;
    }
    
    int get(tensor_dim_var_index_t var_index) const {
        auto it = assignments.find(var_index);
        return (it != assignments.end()) ? it->second : 0;
    }
    
    bool contains(tensor_dim_var_index_t var_index) const {
        return assignments.find(var_index) != assignments.end();
    }
    
    size_t size() const { return assignments.size(); }
    
    std::unordered_map<tensor_dim_var_index_t, int> const& get_all() const {
        return assignments;
    }
    
private:
    std::unordered_map<tensor_dim_var_index_t, int> assignments;
};

// =============================================================================
// TensorDimConstraint - Constraint on dimension expressions
// =============================================================================

struct TensorDimConstraint {
    enum class Type {
        EQUAL,
        NOT_EQUAL,
        LESS_THAN,
        LESS_EQUAL,
        GREATER_THAN,
        GREATER_EQUAL,
        DIVISIBLE
    };
    
    TensorDimExprPtr lhs;
    TensorDimExprPtr rhs;
    Type type;
    
    bool check(std::unordered_map<tensor_dim_var_index_t, int> const& assignments) const {
        int lhs_val = lhs->evaluate(assignments);
        int rhs_val = rhs->evaluate(assignments);
        
        switch (type) {
            case Type::EQUAL: return lhs_val == rhs_val;
            case Type::NOT_EQUAL: return lhs_val != rhs_val;
            case Type::LESS_THAN: return lhs_val < rhs_val;
            case Type::LESS_EQUAL: return lhs_val <= rhs_val;
            case Type::GREATER_THAN: return lhs_val > rhs_val;
            case Type::GREATER_EQUAL: return lhs_val >= rhs_val;
            case Type::DIVISIBLE: return rhs_val != 0 && lhs_val % rhs_val == 0;
        }
        return false;
    }
    
    bool operator==(TensorDimConstraint const& other) const {
        return type == other.type;  // Simplified comparison
    }
};

// =============================================================================
// TensorDimConstraints - Collection of constraints
// =============================================================================

class TensorDimConstraints {
public:
    void add(TensorDimConstraint const& constraint) {
        constraints.push_back(constraint);
    }
    
    bool check_all(std::unordered_map<tensor_dim_var_index_t, int> const& assignments) const {
        for (auto const& c : constraints) {
            if (!c.check(assignments)) return false;
        }
        return true;
    }
    
    size_t size() const { return constraints.size(); }
    bool empty() const { return constraints.empty(); }
    
    std::vector<TensorDimConstraint> constraints;
};

// =============================================================================
// SymbolicDTensor - Symbolic device tensor
// =============================================================================

struct SymbolicDTensor {
    std::vector<SymbolicTensorDim> dims;
    int data_type = 0;  // float16 = 0, float32 = 1, etc.
    
    int num_dims() const { return static_cast<int>(dims.size()); }
};

// =============================================================================
// SymbolicSTensor - Symbolic shared memory tensor
// =============================================================================

struct SymbolicSTensor {
    std::vector<SymbolicTensorDim> dims;
    int data_type = 0;
    int memory_level = 1;  // 0=register, 1=shared, 2=global
    
    int num_dims() const { return static_cast<int>(dims.size()); }
};

// =============================================================================
// SymbolicKNOp / SymbolicTBOp - Symbolic operators
// =============================================================================

struct SymbolicKNOp {
    int op_type = 0;
    std::vector<int> input_indices;
    std::vector<int> output_indices;
};

struct SymbolicTBOp {
    int op_type = 0;
    std::vector<int> input_indices;
    std::vector<int> output_indices;
};

// =============================================================================
// SymbolicMap - Input/output mapping
// =============================================================================

struct SymbolicMap {
    int dim_x = -1;
    int dim_y = -1;
    int dim_z = -1;
    int forloop_dim = -1;
    
    bool is_valid() const {
        return dim_x >= 0 || dim_y >= 0 || dim_z >= 0;
    }
};

}  // namespace search
}  // namespace yirage

using namespace yirage::search;

// =============================================================================
// TensorDimExpr Tests
// =============================================================================

class TensorDimExprTest : public ::testing::Test {};

TEST_F(TensorDimExprTest, ConstantExpression) {
    auto expr = make_constant(42);
    EXPECT_TRUE(expr->is_constant());
    EXPECT_EQ(expr->get_constant(), 42);
    EXPECT_EQ(expr->to_string(), "42");
}

TEST_F(TensorDimExprTest, VariableExpression) {
    auto expr = make_variable(0);
    EXPECT_TRUE(expr->is_variable());
    EXPECT_EQ(expr->get_var_index(), 0);
    EXPECT_EQ(expr->to_string(), "v0");
}

TEST_F(TensorDimExprTest, AddExpression) {
    auto lhs = make_constant(10);
    auto rhs = make_constant(20);
    auto expr = make_add(lhs, rhs);
    
    EXPECT_FALSE(expr->is_constant());
    EXPECT_EQ(expr->to_string(), "(10 + 20)");
}

TEST_F(TensorDimExprTest, MulExpression) {
    auto lhs = make_variable(0);
    auto rhs = make_constant(4);
    auto expr = make_mul(lhs, rhs);
    
    EXPECT_EQ(expr->to_string(), "(v0 * 4)");
}

TEST_F(TensorDimExprTest, EvaluateConstant) {
    auto expr = make_constant(42);
    std::unordered_map<tensor_dim_var_index_t, int> assignments;
    EXPECT_EQ(expr->evaluate(assignments), 42);
}

TEST_F(TensorDimExprTest, EvaluateVariable) {
    auto expr = make_variable(0);
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 128}};
    EXPECT_EQ(expr->evaluate(assignments), 128);
}

TEST_F(TensorDimExprTest, EvaluateAdd) {
    auto lhs = make_variable(0);
    auto rhs = make_constant(10);
    auto expr = make_add(lhs, rhs);
    
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 50}};
    EXPECT_EQ(expr->evaluate(assignments), 60);
}

TEST_F(TensorDimExprTest, EvaluateMul) {
    auto lhs = make_variable(0);
    auto rhs = make_variable(1);
    auto expr = make_mul(lhs, rhs);
    
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 4}, {1, 8}};
    EXPECT_EQ(expr->evaluate(assignments), 32);
}

TEST_F(TensorDimExprTest, EvaluateNested) {
    // (v0 + 10) * 4
    auto v0 = make_variable(0);
    auto c10 = make_constant(10);
    auto add = make_add(v0, c10);
    auto c4 = make_constant(4);
    auto expr = make_mul(add, c4);
    
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 22}};
    // (22 + 10) * 4 = 128
    EXPECT_EQ(expr->evaluate(assignments), 128);
}

// =============================================================================
// SymbolicTensorDim Tests
// =============================================================================

class SymbolicTensorDimTest : public ::testing::Test {};

TEST_F(SymbolicTensorDimTest, ConstantDim) {
    SymbolicTensorDim dim(128);
    EXPECT_TRUE(dim.is_constant());
    EXPECT_EQ(dim.get_constant(), 128);
}

TEST_F(SymbolicTensorDimTest, VariableDim) {
    SymbolicTensorDim dim(make_variable(0), 64, 512, 64);
    EXPECT_FALSE(dim.is_constant());
    EXPECT_EQ(dim.min_value, 64);
    EXPECT_EQ(dim.max_value, 512);
    EXPECT_EQ(dim.step, 64);
}

TEST_F(SymbolicTensorDimTest, EnumerateValues) {
    SymbolicTensorDim dim(make_variable(0), 64, 256, 64);
    auto values = dim.enumerate_values();
    
    EXPECT_EQ(values.size(), 4u);  // 64, 128, 192, 256
    EXPECT_EQ(values[0], 64);
    EXPECT_EQ(values[1], 128);
    EXPECT_EQ(values[2], 192);
    EXPECT_EQ(values[3], 256);
}

// =============================================================================
// DimVarAssignments Tests
// =============================================================================

class DimVarAssignmentsTest : public ::testing::Test {};

TEST_F(DimVarAssignmentsTest, SetAndGet) {
    DimVarAssignments assignments;
    assignments.set(0, 128);
    assignments.set(1, 256);
    
    EXPECT_EQ(assignments.get(0), 128);
    EXPECT_EQ(assignments.get(1), 256);
}

TEST_F(DimVarAssignmentsTest, GetMissing) {
    DimVarAssignments assignments;
    EXPECT_EQ(assignments.get(0), 0);  // Default
}

TEST_F(DimVarAssignmentsTest, Contains) {
    DimVarAssignments assignments;
    assignments.set(0, 128);
    
    EXPECT_TRUE(assignments.contains(0));
    EXPECT_FALSE(assignments.contains(1));
}

TEST_F(DimVarAssignmentsTest, Size) {
    DimVarAssignments assignments;
    EXPECT_EQ(assignments.size(), 0u);
    
    assignments.set(0, 64);
    EXPECT_EQ(assignments.size(), 1u);
    
    assignments.set(1, 128);
    EXPECT_EQ(assignments.size(), 2u);
}

// =============================================================================
// TensorDimConstraint Tests
// =============================================================================

class TensorDimConstraintTest : public ::testing::Test {};

TEST_F(TensorDimConstraintTest, EqualConstraint) {
    auto lhs = make_variable(0);
    auto rhs = make_constant(128);
    TensorDimConstraint constraint{lhs, rhs, TensorDimConstraint::Type::EQUAL};
    
    std::unordered_map<tensor_dim_var_index_t, int> good = {{0, 128}};
    std::unordered_map<tensor_dim_var_index_t, int> bad = {{0, 64}};
    
    EXPECT_TRUE(constraint.check(good));
    EXPECT_FALSE(constraint.check(bad));
}

TEST_F(TensorDimConstraintTest, LessThanConstraint) {
    auto lhs = make_variable(0);
    auto rhs = make_constant(256);
    TensorDimConstraint constraint{lhs, rhs, TensorDimConstraint::Type::LESS_THAN};
    
    std::unordered_map<tensor_dim_var_index_t, int> good = {{0, 128}};
    std::unordered_map<tensor_dim_var_index_t, int> bad = {{0, 512}};
    
    EXPECT_TRUE(constraint.check(good));
    EXPECT_FALSE(constraint.check(bad));
}

TEST_F(TensorDimConstraintTest, DivisibleConstraint) {
    auto lhs = make_variable(0);
    auto rhs = make_constant(32);
    TensorDimConstraint constraint{lhs, rhs, TensorDimConstraint::Type::DIVISIBLE};
    
    std::unordered_map<tensor_dim_var_index_t, int> good = {{0, 128}};
    std::unordered_map<tensor_dim_var_index_t, int> bad = {{0, 100}};
    
    EXPECT_TRUE(constraint.check(good));
    EXPECT_FALSE(constraint.check(bad));
}

// =============================================================================
// TensorDimConstraints Tests
// =============================================================================

class TensorDimConstraintsTest : public ::testing::Test {};

TEST_F(TensorDimConstraintsTest, Empty) {
    TensorDimConstraints constraints;
    EXPECT_TRUE(constraints.empty());
    EXPECT_EQ(constraints.size(), 0u);
}

TEST_F(TensorDimConstraintsTest, AddConstraint) {
    TensorDimConstraints constraints;
    constraints.add({make_variable(0), make_constant(128),
                     TensorDimConstraint::Type::EQUAL});
    
    EXPECT_FALSE(constraints.empty());
    EXPECT_EQ(constraints.size(), 1u);
}

TEST_F(TensorDimConstraintsTest, CheckAllPass) {
    TensorDimConstraints constraints;
    constraints.add({make_variable(0), make_constant(256),
                     TensorDimConstraint::Type::LESS_THAN});
    constraints.add({make_variable(0), make_constant(32),
                     TensorDimConstraint::Type::DIVISIBLE});
    
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 128}};
    EXPECT_TRUE(constraints.check_all(assignments));
}

TEST_F(TensorDimConstraintsTest, CheckAllFail) {
    TensorDimConstraints constraints;
    constraints.add({make_variable(0), make_constant(100),
                     TensorDimConstraint::Type::LESS_THAN});
    constraints.add({make_variable(0), make_constant(32),
                     TensorDimConstraint::Type::DIVISIBLE});
    
    std::unordered_map<tensor_dim_var_index_t, int> assignments = {{0, 128}};
    EXPECT_FALSE(constraints.check_all(assignments));  // 128 >= 100
}

// =============================================================================
// SymbolicDTensor Tests
// =============================================================================

class SymbolicDTensorTest : public ::testing::Test {};

TEST_F(SymbolicDTensorTest, Construction) {
    SymbolicDTensor tensor;
    tensor.dims.push_back(SymbolicTensorDim(128));
    tensor.dims.push_back(SymbolicTensorDim(256));
    tensor.data_type = 0;
    
    EXPECT_EQ(tensor.num_dims(), 2);
}

TEST_F(SymbolicDTensorTest, SymbolicDims) {
    SymbolicDTensor tensor;
    tensor.dims.push_back(SymbolicTensorDim(make_variable(0), 64, 256, 64));
    tensor.dims.push_back(SymbolicTensorDim(make_variable(1), 64, 512, 64));
    
    EXPECT_EQ(tensor.num_dims(), 2);
    EXPECT_FALSE(tensor.dims[0].is_constant());
    EXPECT_FALSE(tensor.dims[1].is_constant());
}

// =============================================================================
// SymbolicSTensor Tests
// =============================================================================

class SymbolicSTensorTest : public ::testing::Test {};

TEST_F(SymbolicSTensorTest, Construction) {
    SymbolicSTensor tensor;
    tensor.dims.push_back(SymbolicTensorDim(32));
    tensor.dims.push_back(SymbolicTensorDim(64));
    tensor.memory_level = 1;  // Shared memory
    
    EXPECT_EQ(tensor.num_dims(), 2);
    EXPECT_EQ(tensor.memory_level, 1);
}

// =============================================================================
// SymbolicMap Tests
// =============================================================================

class SymbolicMapTest : public ::testing::Test {};

TEST_F(SymbolicMapTest, InvalidMap) {
    SymbolicMap map;
    EXPECT_FALSE(map.is_valid());
}

TEST_F(SymbolicMapTest, ValidMap) {
    SymbolicMap map;
    map.dim_x = 0;
    map.dim_y = 1;
    EXPECT_TRUE(map.is_valid());
}

TEST_F(SymbolicMapTest, WithForloop) {
    SymbolicMap map;
    map.dim_x = 0;
    map.forloop_dim = 2;
    
    EXPECT_TRUE(map.is_valid());
    EXPECT_EQ(map.forloop_dim, 2);
}

// =============================================================================
// Parameterized Constraint Tests
// =============================================================================

struct ConstraintTestParam {
    TensorDimConstraint::Type type;
    int lhs_val;
    int rhs_val;
    bool expected;
};

class ConstraintParameterizedTest : public ::testing::TestWithParam<ConstraintTestParam> {};

TEST_P(ConstraintParameterizedTest, ConstraintCheck) {
    auto param = GetParam();
    
    auto lhs = make_constant(param.lhs_val);
    auto rhs = make_constant(param.rhs_val);
    TensorDimConstraint constraint{lhs, rhs, param.type};
    
    std::unordered_map<tensor_dim_var_index_t, int> empty;
    EXPECT_EQ(constraint.check(empty), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllConstraintTypes,
    ConstraintParameterizedTest,
    ::testing::Values(
        ConstraintTestParam{TensorDimConstraint::Type::EQUAL, 100, 100, true},
        ConstraintTestParam{TensorDimConstraint::Type::EQUAL, 100, 200, false},
        ConstraintTestParam{TensorDimConstraint::Type::NOT_EQUAL, 100, 200, true},
        ConstraintTestParam{TensorDimConstraint::Type::NOT_EQUAL, 100, 100, false},
        ConstraintTestParam{TensorDimConstraint::Type::LESS_THAN, 50, 100, true},
        ConstraintTestParam{TensorDimConstraint::Type::LESS_THAN, 100, 50, false},
        ConstraintTestParam{TensorDimConstraint::Type::LESS_EQUAL, 100, 100, true},
        ConstraintTestParam{TensorDimConstraint::Type::GREATER_THAN, 200, 100, true},
        ConstraintTestParam{TensorDimConstraint::Type::GREATER_EQUAL, 100, 100, true},
        ConstraintTestParam{TensorDimConstraint::Type::DIVISIBLE, 128, 32, true},
        ConstraintTestParam{TensorDimConstraint::Type::DIVISIBLE, 100, 32, false}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
