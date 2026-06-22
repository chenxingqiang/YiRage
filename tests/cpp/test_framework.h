/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_framework.h
 * @brief Lightweight C++ testing framework for YiRage
 * 
 * A simple, self-contained testing framework inspired by GoogleTest.
 * Supports test cases, test suites, assertions, and test fixtures.
 */

#ifndef YIRAGE_TEST_FRAMEWORK_H
#define YIRAGE_TEST_FRAMEWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <sstream>
#include <memory>
#include <map>
#include <iomanip>

namespace yirage {
namespace test {

// =============================================================================
// Color Output Support
// =============================================================================

struct Color {
    static constexpr const char* Reset = "\033[0m";
    static constexpr const char* Red = "\033[31m";
    static constexpr const char* Green = "\033[32m";
    static constexpr const char* Yellow = "\033[33m";
    static constexpr const char* Blue = "\033[34m";
    static constexpr const char* Magenta = "\033[35m";
    static constexpr const char* Cyan = "\033[36m";
    static constexpr const char* Bold = "\033[1m";
};

// =============================================================================
// Test Result
// =============================================================================

struct TestResult {
    bool passed = true;
    std::string message;
    std::string file;
    int line = 0;
    double duration_ms = 0.0;
};

// =============================================================================
// Test Case
// =============================================================================

class TestCase {
public:
    std::string suite_name;
    std::string test_name;
    std::function<TestResult()> func;
    bool skip = false;
    std::string skip_reason;
    std::vector<std::string> tags;
    
    TestCase(const std::string& suite, const std::string& name, 
             std::function<TestResult()> f)
        : suite_name(suite), test_name(name), func(f) {}
    
    std::string full_name() const {
        return suite_name + "." + test_name;
    }
};

// =============================================================================
// Test Registry (Singleton)
// =============================================================================

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }
    
    void register_test(std::unique_ptr<TestCase> test) {
        tests_.push_back(std::move(test));
    }
    
    void register_setup(const std::string& suite, std::function<void()> func) {
        setup_funcs_[suite] = func;
    }
    
    void register_teardown(const std::string& suite, std::function<void()> func) {
        teardown_funcs_[suite] = func;
    }
    
    const std::vector<std::unique_ptr<TestCase>>& tests() const {
        return tests_;
    }
    
    std::function<void()> get_setup(const std::string& suite) const {
        auto it = setup_funcs_.find(suite);
        return it != setup_funcs_.end() ? it->second : nullptr;
    }
    
    std::function<void()> get_teardown(const std::string& suite) const {
        auto it = teardown_funcs_.find(suite);
        return it != teardown_funcs_.end() ? it->second : nullptr;
    }

private:
    TestRegistry() = default;
    std::vector<std::unique_ptr<TestCase>> tests_;
    std::map<std::string, std::function<void()>> setup_funcs_;
    std::map<std::string, std::function<void()>> teardown_funcs_;
};

// =============================================================================
// Test Runner
// =============================================================================

class TestRunner {
public:
    struct Config {
        bool verbose = false;
        bool color = true;
        std::string filter;  // Only run tests matching this pattern
        std::string exclude;  // Exclude tests matching this pattern
        bool list_only = false;
        bool shuffle = false;
        int repeat = 1;
    };
    
    TestRunner(Config config = {}) : config_(config) {}
    
    int run() {
        auto& registry = TestRegistry::instance();
        const auto& tests = registry.tests();
        
        if (config_.list_only) {
            list_tests(tests);
            return 0;
        }
        
        print_header(tests.size());
        
        int passed = 0;
        int failed = 0;
        int skipped = 0;
        double total_time = 0.0;
        
        std::string current_suite;
        
        for (const auto& test : tests) {
            // Filter check
            if (!matches_filter(test->full_name())) {
                continue;
            }
            
            // Suite change
            if (test->suite_name != current_suite) {
                current_suite = test->suite_name;
                print_suite_header(current_suite);
                
                // Run setup if exists
                auto setup = registry.get_setup(current_suite);
                if (setup) {
                    try {
                        setup();
                    } catch (const std::exception& e) {
                        std::cerr << Color::Red << "Suite setup failed: " 
                                  << e.what() << Color::Reset << std::endl;
                    }
                }
            }
            
            // Skip check
            if (test->skip) {
                print_skip(test->test_name, test->skip_reason);
                ++skipped;
                continue;
            }
            
            // Run test
            auto start = std::chrono::high_resolution_clock::now();
            TestResult result;
            
            try {
                result = test->func();
            } catch (const std::exception& e) {
                result.passed = false;
                result.message = std::string("Exception: ") + e.what();
            } catch (...) {
                result.passed = false;
                result.message = "Unknown exception";
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_time += result.duration_ms;
            
            if (result.passed) {
                print_pass(test->test_name, result.duration_ms);
                ++passed;
            } else {
                print_fail(test->test_name, result);
                ++failed;
            }
        }
        
        // Run teardowns
        for (const auto& test : tests) {
            auto teardown = registry.get_teardown(test->suite_name);
            if (teardown) {
                try {
                    teardown();
                } catch (...) {}
            }
        }
        
        print_summary(passed, failed, skipped, total_time);
        
        return failed > 0 ? 1 : 0;
    }

private:
    Config config_;
    
    bool matches_filter(const std::string& name) const {
        if (!config_.filter.empty() && 
            name.find(config_.filter) == std::string::npos) {
            return false;
        }
        if (!config_.exclude.empty() && 
            name.find(config_.exclude) != std::string::npos) {
            return false;
        }
        return true;
    }
    
    void list_tests(const std::vector<std::unique_ptr<TestCase>>& tests) const {
        std::cout << "Available tests:\n";
        for (const auto& test : tests) {
            std::cout << "  " << test->full_name() << "\n";
        }
    }
    
    void print_header(size_t count) const {
        std::cout << Color::Bold << Color::Cyan
                  << "\n╔═══════════════════════════════════════════════════════════════╗\n"
                  << "║              YiRage C++ Test Suite                             ║\n"
                  << "║              Running " << std::setw(4) << count << " test(s)                              ║\n"
                  << "╚═══════════════════════════════════════════════════════════════╝\n"
                  << Color::Reset << std::endl;
    }
    
    void print_suite_header(const std::string& suite) const {
        std::cout << Color::Bold << Color::Blue
                  << "\n┌───────────────────────────────────────────────────────────────┐\n"
                  << "│ Suite: " << std::left << std::setw(55) << suite << " │\n"
                  << "└───────────────────────────────────────────────────────────────┘"
                  << Color::Reset << std::endl;
    }
    
    void print_pass(const std::string& name, double ms) const {
        std::cout << Color::Green << "  ✓ " << Color::Reset 
                  << std::left << std::setw(50) << name
                  << Color::Cyan << " (" << std::fixed << std::setprecision(2) << ms << " ms)"
                  << Color::Reset << std::endl;
    }
    
    void print_fail(const std::string& name, const TestResult& result) const {
        std::cout << Color::Red << "  ✗ " << Color::Reset << name << std::endl;
        if (!result.file.empty()) {
            std::cout << Color::Yellow << "      at " << result.file << ":" << result.line << Color::Reset << std::endl;
        }
        if (!result.message.empty()) {
            std::cout << Color::Yellow << "      " << result.message << Color::Reset << std::endl;
        }
    }
    
    void print_skip(const std::string& name, const std::string& reason) const {
        std::cout << Color::Yellow << "  ⊘ " << Color::Reset << name;
        if (!reason.empty()) {
            std::cout << " (" << reason << ")";
        }
        std::cout << std::endl;
    }
    
    void print_summary(int passed, int failed, int skipped, double total_ms) const {
        std::cout << Color::Bold << Color::Cyan
                  << "\n╔═══════════════════════════════════════════════════════════════╗\n"
                  << "║                         Summary                                ║\n"
                  << "╠═══════════════════════════════════════════════════════════════╣\n";
        
        std::cout << "║  " << Color::Green << "Passed:  " << std::setw(5) << passed << Color::Cyan << std::setw(48) << " ║\n";
        std::cout << "║  " << Color::Red << "Failed:  " << std::setw(5) << failed << Color::Cyan << std::setw(48) << " ║\n";
        std::cout << "║  " << Color::Yellow << "Skipped: " << std::setw(5) << skipped << Color::Cyan << std::setw(48) << " ║\n";
        std::cout << "║  Total Time: " << std::fixed << std::setprecision(2) << total_ms << " ms" 
                  << std::setw(45 - std::to_string(static_cast<int>(total_ms)).length()) << " ║\n";
        
        std::cout << "╚═══════════════════════════════════════════════════════════════╝"
                  << Color::Reset << std::endl;
        
        if (failed == 0) {
            std::cout << Color::Bold << Color::Green << "\n✓ All tests passed!\n" << Color::Reset << std::endl;
        } else {
            std::cout << Color::Bold << Color::Red << "\n✗ " << failed << " test(s) failed.\n" << Color::Reset << std::endl;
        }
    }
};

// =============================================================================
// Assertion Macros
// =============================================================================

#define YIRAGE_TEST_RESULT_PASS() \
    ::yirage::test::TestResult{true, "", "", 0, 0.0}

#define YIRAGE_TEST_RESULT_FAIL(msg) \
    ::yirage::test::TestResult{false, (msg), __FILE__, __LINE__, 0.0}

#define EXPECT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            return YIRAGE_TEST_RESULT_FAIL("Expected true: " #cond); \
        } \
    } while (0)

#define EXPECT_FALSE(cond) \
    do { \
        if (cond) { \
            return YIRAGE_TEST_RESULT_FAIL("Expected false: " #cond); \
        } \
    } while (0)

#define EXPECT_EQ(a, b) \
    do { \
        if (!((a) == (b))) { \
            std::ostringstream oss; \
            oss << "Expected " #a " == " #b ", got " << (a) << " != " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            std::ostringstream oss; \
            oss << "Expected " #a " != " #b ", got " << (a) << " == " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_GT(a, b) \
    do { \
        if (!((a) > (b))) { \
            std::ostringstream oss; \
            oss << "Expected " #a " > " #b ", got " << (a) << " <= " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_GE(a, b) \
    do { \
        if (!((a) >= (b))) { \
            std::ostringstream oss; \
            oss << "Expected " #a " >= " #b ", got " << (a) << " < " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_LT(a, b) \
    do { \
        if (!((a) < (b))) { \
            std::ostringstream oss; \
            oss << "Expected " #a " < " #b ", got " << (a) << " >= " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_LE(a, b) \
    do { \
        if (!((a) <= (b))) { \
            std::ostringstream oss; \
            oss << "Expected " #a " <= " #b ", got " << (a) << " > " << (b); \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_NULL(ptr) \
    do { \
        if ((ptr) != nullptr) { \
            return YIRAGE_TEST_RESULT_FAIL("Expected nullptr: " #ptr); \
        } \
    } while (0)

#define EXPECT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == nullptr) { \
            return YIRAGE_TEST_RESULT_FAIL("Expected not nullptr: " #ptr); \
        } \
    } while (0)

#define EXPECT_NEAR(a, b, eps) \
    do { \
        auto diff = std::abs((a) - (b)); \
        if (diff > (eps)) { \
            std::ostringstream oss; \
            oss << "Expected |" #a " - " #b "| <= " #eps ", got " << diff; \
            return YIRAGE_TEST_RESULT_FAIL(oss.str()); \
        } \
    } while (0)

#define EXPECT_THROWS(expr) \
    do { \
        bool threw = false; \
        try { expr; } catch (...) { threw = true; } \
        if (!threw) { \
            return YIRAGE_TEST_RESULT_FAIL("Expected exception: " #expr); \
        } \
    } while (0)

#define EXPECT_NO_THROW(expr) \
    do { \
        try { expr; } catch (...) { \
            return YIRAGE_TEST_RESULT_FAIL("Unexpected exception: " #expr); \
        } \
    } while (0)

#define SKIP_IF(cond, reason) \
    do { \
        if (cond) { \
            return ::yirage::test::TestResult{true, "", "", 0, 0.0}; \
        } \
    } while (0)

// =============================================================================
// Test Registration Macros
// =============================================================================

#define TEST(suite, name) \
    static ::yirage::test::TestResult test_##suite##_##name##_impl(); \
    static struct TestRegister_##suite##_##name { \
        TestRegister_##suite##_##name() { \
            ::yirage::test::TestRegistry::instance().register_test( \
                std::make_unique<::yirage::test::TestCase>( \
                    #suite, #name, test_##suite##_##name##_impl)); \
        } \
    } test_register_##suite##_##name; \
    static ::yirage::test::TestResult test_##suite##_##name##_impl()

#define TEST_F(fixture, name) \
    static ::yirage::test::TestResult test_##fixture##_##name##_impl(); \
    static struct TestRegister_##fixture##_##name { \
        TestRegister_##fixture##_##name() { \
            ::yirage::test::TestRegistry::instance().register_test( \
                std::make_unique<::yirage::test::TestCase>( \
                    #fixture, #name, test_##fixture##_##name##_impl)); \
        } \
    } test_register_##fixture##_##name; \
    static ::yirage::test::TestResult test_##fixture##_##name##_impl()

#define YIRAGE_TEST_MAIN() \
    int main(int argc, char** argv) { \
        ::yirage::test::TestRunner::Config config; \
        for (int i = 1; i < argc; ++i) { \
            std::string arg = argv[i]; \
            if (arg == "--verbose" || arg == "-v") config.verbose = true; \
            else if (arg == "--no-color") config.color = false; \
            else if (arg == "--list") config.list_only = true; \
            else if (arg.rfind("--filter=", 0) == 0) config.filter = arg.substr(9); \
            else if (arg.rfind("--exclude=", 0) == 0) config.exclude = arg.substr(10); \
        } \
        return ::yirage::test::TestRunner(config).run(); \
    }

}  // namespace test
}  // namespace yirage

#endif  // YIRAGE_TEST_FRAMEWORK_H
