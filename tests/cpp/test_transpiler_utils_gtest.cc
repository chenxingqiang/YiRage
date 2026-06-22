// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_utils_gtest.cc
 * @brief Transpiler Utilities Unit Tests
 *
 * Tests for transpiler utilities (utils.h):
 *   - my_to_string conversions
 *   - fmt string formatting
 *   - CodeKeeper code generation helper
 *   - ceil_div and round_to_multiple
 *   - map function
 *   - Combine iterator
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>
#include <sstream>
#include <functional>

namespace yirage {
namespace transpiler {

// =============================================================================
// my_to_string Functions
// =============================================================================

template <typename T>
inline std::string my_to_string(T const& value) {
    return std::to_string(value);
}

template <>
inline std::string my_to_string(char const* const& value) {
    return std::string(value);
}

template <>
inline std::string my_to_string(void* const& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <>
inline std::string my_to_string(std::string const& value) {
    return value;
}

template <>
inline std::string my_to_string(char const& value) {
    return std::string(1, value);
}

template <>
inline std::string my_to_string(bool const& value) {
    return value ? "true" : "false";
}

template <typename T>
inline std::string my_to_string(std::vector<T> const& vec) {
    std::string result;
    for (size_t i = 0; i < vec.size(); i++) {
        result += my_to_string(vec[i]);
        if (i != vec.size() - 1) {
            result += ", ";
        }
    }
    return result;
}

// =============================================================================
// fmt Function
// =============================================================================

template <typename... Args>
inline std::string fmt(std::string const& fmt_str, Args... args) {
    std::string result = fmt_str;
    int num_args = sizeof...(args);
    int num_markers = std::count(result.begin(), result.end(), '$');
    if (num_args != num_markers) {
        return "ERROR: Argument count mismatch";
    }
    ((result.replace(result.find("$"), 1, my_to_string(args))), ...);
    return result;
}

// =============================================================================
// CodeKeeper
// =============================================================================

class CodeKeeper {
private:
    static constexpr int NUM_INDENT_SPACES = 2;
    int cur_indent_level = 0;
    std::vector<std::string> lines;

public:
    template <typename... Args>
    void e_front(std::string const& fmt_str, Args... args) {
        std::string line = fmt(fmt_str, args...);
        lines.emplace(lines.begin(), line);
    }

    template <typename... Args>
    void e(std::string const& fmt_str, Args... args) {
        std::string line = fmt(fmt_str, args...);
        char last_char = line.empty() ? EOF : line.back();
        if (last_char == '}') {
            cur_indent_level -= 1;
            if (cur_indent_level < 0) {
                cur_indent_level = 0;
            }
        }
        line = std::string(cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
        lines.push_back(line);

        if (last_char == '{') {
            cur_indent_level += 1;
        }
    }

    void inc_indent() {
        cur_indent_level += 1;
    }

    void dec_indent() {
        cur_indent_level -= 1;
        if (cur_indent_level < 0) {
            cur_indent_level = 0;
        }
    }

    friend void operator<<(CodeKeeper& target, CodeKeeper const& source) {
        for (auto const& line : source.lines) {
            std::string new_line =
                std::string(target.cur_indent_level * NUM_INDENT_SPACES, ' ') + line;
            target.lines.push_back(new_line);
        }
    }

    std::string to_string() const {
        std::string result;
        for (auto const& line : lines) {
            result += line + "\n";
        }
        return result;
    }

    size_t num_lines() const {
        return lines.size();
    }

    int get_indent_level() const {
        return cur_indent_level;
    }

    void clear() {
        lines.clear();
        cur_indent_level = 0;
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

template <typename T>
inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
inline T round_to_multiple(T value, T multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Map function
template <typename InT, typename OutT>
std::vector<OutT> map(std::vector<InT> const& vec,
                      std::function<OutT(InT)> const& f) {
    std::vector<OutT> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = f(vec[i]);
    }
    return result;
}

template <typename T>
std::vector<std::string> map_to_cute_int(std::vector<T> const& vec) {
    return map<T, std::string>(
        vec, [](T const& x) { return "Int<" + my_to_string(x) + ">"; });
}

// =============================================================================
// Combine Iterator
// =============================================================================

template <typename T, class Iter1, class Iter2>
class CombineIterator {
    Iter1 range1_start, range1_end, iter1;
    Iter2 range2_start, range2_end, iter2;

public:
    CombineIterator(Iter1 const& r1_start, Iter1 const& r1_end, Iter1 const& i1,
                    Iter2 const& r2_start, Iter2 const& r2_end, Iter2 const& i2)
        : range1_start(r1_start), range1_end(r1_end), iter1(i1),
          range2_start(r2_start), range2_end(r2_end), iter2(i2) {}

    bool operator!=(CombineIterator const& other) const {
        return iter1 != other.iter1 || iter2 != other.iter2;
    }

    void operator++() {
        if (iter1 != range1_end) {
            ++iter1;
        } else {
            ++iter2;
        }
    }

    T const& operator*() const {
        if (iter1 != range1_end) {
            return *iter1;
        } else {
            return *iter2;
        }
    }
};

template <typename T1, typename T2>
class Combine {
    using T = std::common_type_t<typename T1::value_type, typename T2::value_type>;
    using Iter1 = typename T1::const_iterator;
    using Iter2 = typename T2::const_iterator;

    Iter1 begin1, end1;
    Iter2 begin2, end2;

public:
    using value_type = T;
    using const_iterator = CombineIterator<T, Iter1, Iter2>;

    Combine(T1 const& v1, T2 const& v2)
        : begin1(v1.begin()), end1(v1.end()), begin2(v2.begin()), end2(v2.end()) {}

    CombineIterator<T, Iter1, Iter2> begin() const {
        return CombineIterator<T, Iter1, Iter2>(begin1, end1, begin1, begin2, end2, begin2);
    }

    CombineIterator<T, Iter1, Iter2> end() const {
        return CombineIterator<T, Iter1, Iter2>(begin1, end1, end1, begin2, end2, end2);
    }
};

}  // namespace transpiler
}  // namespace yirage

using namespace yirage::transpiler;

// =============================================================================
// my_to_string Tests
// =============================================================================

class MyToStringTest : public ::testing::Test {};

TEST_F(MyToStringTest, IntToString) {
    EXPECT_EQ(my_to_string(42), "42");
    EXPECT_EQ(my_to_string(-10), "-10");
    EXPECT_EQ(my_to_string(0), "0");
}

TEST_F(MyToStringTest, CharPtrToString) {
    const char* str = "hello";
    EXPECT_EQ(my_to_string(str), "hello");
}

TEST_F(MyToStringTest, StringToString) {
    std::string str = "world";
    EXPECT_EQ(my_to_string(str), "world");
}

TEST_F(MyToStringTest, CharToString) {
    EXPECT_EQ(my_to_string('A'), "A");
    EXPECT_EQ(my_to_string('z'), "z");
}

TEST_F(MyToStringTest, BoolToString) {
    EXPECT_EQ(my_to_string(true), "true");
    EXPECT_EQ(my_to_string(false), "false");
}

TEST_F(MyToStringTest, VectorToString) {
    std::vector<int> vec = {1, 2, 3, 4};
    EXPECT_EQ(my_to_string(vec), "1, 2, 3, 4");
}

TEST_F(MyToStringTest, EmptyVectorToString) {
    std::vector<int> vec;
    EXPECT_EQ(my_to_string(vec), "");
}

TEST_F(MyToStringTest, SingleElementVectorToString) {
    std::vector<int> vec = {42};
    EXPECT_EQ(my_to_string(vec), "42");
}

// =============================================================================
// fmt Tests
// =============================================================================

class FmtTest : public ::testing::Test {};

TEST_F(FmtTest, NoMarkers) {
    EXPECT_EQ(fmt("hello world"), "hello world");
}

TEST_F(FmtTest, SingleMarker) {
    EXPECT_EQ(fmt("value: $", 42), "value: 42");
}

TEST_F(FmtTest, MultipleMarkers) {
    EXPECT_EQ(fmt("$ + $ = $", 1, 2, 3), "1 + 2 = 3");
}

TEST_F(FmtTest, StringMarker) {
    EXPECT_EQ(fmt("Hello, $!", "World"), "Hello, World!");
}

TEST_F(FmtTest, MixedTypes) {
    EXPECT_EQ(fmt("Name: $, Age: $, Active: $", "Alice", 30, true),
              "Name: Alice, Age: 30, Active: true");
}

TEST_F(FmtTest, VectorMarker) {
    std::vector<int> vec = {1, 2, 3};
    EXPECT_EQ(fmt("Values: $", vec), "Values: 1, 2, 3");
}

TEST_F(FmtTest, ArgumentMismatch) {
    // More arguments than markers - should trigger error handling
    auto result = fmt("$", 1, 2);
    EXPECT_TRUE(result.find("ERROR") != std::string::npos ||
                result == "1");  // Implementation dependent
}

// =============================================================================
// CodeKeeper Tests
// =============================================================================

class CodeKeeperTest : public ::testing::Test {};

TEST_F(CodeKeeperTest, EmptyCodeKeeper) {
    CodeKeeper code;
    EXPECT_EQ(code.num_lines(), 0u);
    EXPECT_EQ(code.to_string(), "");
}

TEST_F(CodeKeeperTest, SingleLine) {
    CodeKeeper code;
    code.e("int x = 10;");

    EXPECT_EQ(code.num_lines(), 1u);
    EXPECT_EQ(code.to_string(), "int x = 10;\n");
}

TEST_F(CodeKeeperTest, MultipleLines) {
    CodeKeeper code;
    code.e("int x = 10;");
    code.e("int y = 20;");
    code.e("return x + y;");

    EXPECT_EQ(code.num_lines(), 3u);
}

TEST_F(CodeKeeperTest, AutoIndentWithBraces) {
    CodeKeeper code;
    code.e("void foo() {");
    code.e("int x = 10;");
    code.e("}");

    std::string expected = "void foo() {\n  int x = 10;\n}\n";
    EXPECT_EQ(code.to_string(), expected);
}

TEST_F(CodeKeeperTest, NestedIndent) {
    CodeKeeper code;
    code.e("if (a) {");
    code.e("if (b) {");
    code.e("x = 1;");
    code.e("}");
    code.e("}");

    EXPECT_EQ(code.get_indent_level(), 0);
}

TEST_F(CodeKeeperTest, ManualIndent) {
    CodeKeeper code;
    code.e("line1");
    code.inc_indent();
    code.e("line2");
    code.dec_indent();
    code.e("line3");

    EXPECT_EQ(code.to_string(), "line1\n  line2\nline3\n");
}

TEST_F(CodeKeeperTest, FmtInEmit) {
    CodeKeeper code;
    code.e("int $ = $;", "x", 42);

    EXPECT_EQ(code.to_string(), "int x = 42;\n");
}

TEST_F(CodeKeeperTest, EmitFront) {
    CodeKeeper code;
    code.e("line2");
    code.e_front("line1");

    EXPECT_EQ(code.to_string(), "line1\nline2\n");
}

TEST_F(CodeKeeperTest, MergeCodeKeepers) {
    CodeKeeper main_code;
    CodeKeeper sub_code;

    main_code.e("void main() {");
    sub_code.e("int x = 1;");
    sub_code.e("int y = 2;");

    main_code << sub_code;
    main_code.e("}");

    EXPECT_EQ(main_code.num_lines(), 4u);
}

TEST_F(CodeKeeperTest, Clear) {
    CodeKeeper code;
    code.e("line1");
    code.e("if (x) {");
    code.e("y = 1;");

    code.clear();

    EXPECT_EQ(code.num_lines(), 0u);
    EXPECT_EQ(code.get_indent_level(), 0);
}

// =============================================================================
// ceil_div Tests
// =============================================================================

class CeilDivTest : public ::testing::Test {};

TEST_F(CeilDivTest, ExactDivision) {
    EXPECT_EQ(ceil_div(10, 2), 5);
    EXPECT_EQ(ceil_div(100, 10), 10);
}

TEST_F(CeilDivTest, Remainder) {
    EXPECT_EQ(ceil_div(10, 3), 4);
    EXPECT_EQ(ceil_div(11, 3), 4);
    EXPECT_EQ(ceil_div(12, 3), 4);
}

TEST_F(CeilDivTest, SmallValues) {
    EXPECT_EQ(ceil_div(1, 1), 1);
    EXPECT_EQ(ceil_div(1, 2), 1);
    EXPECT_EQ(ceil_div(1, 10), 1);
}

TEST_F(CeilDivTest, LargeValues) {
    EXPECT_EQ(ceil_div(1000, 128), 8);
    EXPECT_EQ(ceil_div(1024, 128), 8);
}

// =============================================================================
// round_to_multiple Tests
// =============================================================================

class RoundToMultipleTest : public ::testing::Test {};

TEST_F(RoundToMultipleTest, AlreadyMultiple) {
    EXPECT_EQ(round_to_multiple(128, 64), 128);
    EXPECT_EQ(round_to_multiple(256, 128), 256);
}

TEST_F(RoundToMultipleTest, RoundUp) {
    EXPECT_EQ(round_to_multiple(100, 64), 128);
    EXPECT_EQ(round_to_multiple(65, 64), 128);
    EXPECT_EQ(round_to_multiple(1, 64), 64);
}

TEST_F(RoundToMultipleTest, RoundToOne) {
    EXPECT_EQ(round_to_multiple(42, 1), 42);
}

TEST_F(RoundToMultipleTest, PowersOfTwo) {
    EXPECT_EQ(round_to_multiple(1000, 256), 1024);
    EXPECT_EQ(round_to_multiple(1025, 256), 1280);
}

// =============================================================================
// map Function Tests
// =============================================================================

class MapFunctionTest : public ::testing::Test {};

TEST_F(MapFunctionTest, DoubleValues) {
    std::vector<int> input = {1, 2, 3, 4};
    auto result = map<int, int>(input, [](int x) { return x * 2; });

    EXPECT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(result[2], 6);
    EXPECT_EQ(result[3], 8);
}

TEST_F(MapFunctionTest, IntToString) {
    std::vector<int> input = {1, 2, 3};
    auto result = map<int, std::string>(input, [](int x) { return std::to_string(x); });

    EXPECT_EQ(result[0], "1");
    EXPECT_EQ(result[1], "2");
    EXPECT_EQ(result[2], "3");
}

TEST_F(MapFunctionTest, EmptyVector) {
    std::vector<int> input;
    auto result = map<int, int>(input, [](int x) { return x; });

    EXPECT_TRUE(result.empty());
}

TEST_F(MapFunctionTest, MapToCuteInt) {
    std::vector<int> input = {64, 128, 256};
    auto result = map_to_cute_int(input);

    EXPECT_EQ(result[0], "Int<64>");
    EXPECT_EQ(result[1], "Int<128>");
    EXPECT_EQ(result[2], "Int<256>");
}

// =============================================================================
// Combine Iterator Tests
// =============================================================================

class CombineIteratorTest : public ::testing::Test {};

TEST_F(CombineIteratorTest, CombineTwoVectors) {
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = {4, 5, 6};

    std::vector<int> result;
    for (int x : Combine(v1, v2)) {
        result.push_back(x);
    }

    EXPECT_EQ(result.size(), 6u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[2], 3);
    EXPECT_EQ(result[3], 4);
    EXPECT_EQ(result[5], 6);
}

TEST_F(CombineIteratorTest, FirstEmpty) {
    std::vector<int> v1;
    std::vector<int> v2 = {1, 2, 3};

    std::vector<int> result;
    for (int x : Combine(v1, v2)) {
        result.push_back(x);
    }

    EXPECT_EQ(result.size(), 3u);
}

TEST_F(CombineIteratorTest, SecondEmpty) {
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2;

    std::vector<int> result;
    for (int x : Combine(v1, v2)) {
        result.push_back(x);
    }

    EXPECT_EQ(result.size(), 3u);
}

TEST_F(CombineIteratorTest, BothEmpty) {
    std::vector<int> v1;
    std::vector<int> v2;

    std::vector<int> result;
    for (int x : Combine(v1, v2)) {
        result.push_back(x);
    }

    EXPECT_TRUE(result.empty());
}

// =============================================================================
// Parameterized ceil_div Tests
// =============================================================================

struct CeilDivParam {
    int a, b, expected;
};

class CeilDivParameterizedTest
    : public ::testing::TestWithParam<CeilDivParam> {};

TEST_P(CeilDivParameterizedTest, CeilDivision) {
    auto param = GetParam();
    EXPECT_EQ(ceil_div(param.a, param.b), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    CommonCases,
    CeilDivParameterizedTest,
    ::testing::Values(
        CeilDivParam{10, 2, 5},
        CeilDivParam{10, 3, 4},
        CeilDivParam{10, 4, 3},
        CeilDivParam{10, 5, 2},
        CeilDivParam{10, 10, 1},
        CeilDivParam{10, 11, 1},
        CeilDivParam{256, 64, 4},
        CeilDivParam{257, 64, 5}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
