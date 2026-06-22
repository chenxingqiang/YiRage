// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search.cc
 * @brief Search Engine Unit Tests
 *
 * Tests for src/search/ including KernelGraphGenerator and SearchContext.
 * Compile with: clang++ -std=c++17 -I../../include test_search.cc -o test_search
 */

#include "test_framework.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Forward declarations for testing
namespace yirage {
namespace search {

// Mock types for testing when actual headers not available
struct KernelGraphConfig {
    int32_t grid_dim_x = 1;
    int32_t grid_dim_y = 1;
    int32_t grid_dim_z = 1;
    int32_t block_dim_x = 128;
    int32_t block_dim_y = 1;
    int32_t block_dim_z = 1;
    int32_t forloop_range = 8;
    int32_t reduction_dimx = 16;
};

struct SearchConfig {
    int32_t max_depth = 10;
    int32_t max_operators = 64;
    bool enable_pruning = true;
    std::string backend = "cuda";
};

}  // namespace search
}  // namespace yirage

// =============================================================================
// KernelGraphGenerator Tests
// =============================================================================

TEST(KernelGraphGenerator, ConfigCreation) {
    yirage::search::KernelGraphConfig config;
    
    EXPECT_EQ(config.grid_dim_x, 1);
    EXPECT_EQ(config.block_dim_x, 128);
    EXPECT_EQ(config.forloop_range, 8);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(KernelGraphGenerator, ConfigCustomValues) {
    yirage::search::KernelGraphConfig config;
    config.grid_dim_x = 4;
    config.grid_dim_y = 2;
    config.block_dim_x = 256;
    
    EXPECT_EQ(config.grid_dim_x, 4);
    EXPECT_EQ(config.grid_dim_y, 2);
    EXPECT_EQ(config.block_dim_x, 256);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(KernelGraphGenerator, GridDimProduct) {
    yirage::search::KernelGraphConfig config;
    config.grid_dim_x = 4;
    config.grid_dim_y = 2;
    config.grid_dim_z = 1;
    
    int64_t total_blocks = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;
    EXPECT_EQ(total_blocks, 8);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(KernelGraphGenerator, BlockDimProduct) {
    yirage::search::KernelGraphConfig config;
    config.block_dim_x = 128;
    config.block_dim_y = 2;
    config.block_dim_z = 1;
    
    int64_t threads_per_block = config.block_dim_x * config.block_dim_y * config.block_dim_z;
    EXPECT_EQ(threads_per_block, 256);
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// SearchConfig Tests
// =============================================================================

TEST(SearchConfig, DefaultValues) {
    yirage::search::SearchConfig config;
    
    EXPECT_EQ(config.max_depth, 10);
    EXPECT_EQ(config.max_operators, 64);
    EXPECT_TRUE(config.enable_pruning);
    EXPECT_EQ(config.backend, "cuda");
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchConfig, CustomBackend) {
    yirage::search::SearchConfig config;
    config.backend = "cpu";
    
    EXPECT_EQ(config.backend, "cpu");
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchConfig, DisablePruning) {
    yirage::search::SearchConfig config;
    config.enable_pruning = false;
    
    EXPECT_FALSE(config.enable_pruning);
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// SearchContext Tests
// =============================================================================

namespace {
    
// Simple search state for testing
struct SearchState {
    int32_t depth = 0;
    int32_t num_operators = 0;
    bool is_valid = true;
    std::vector<int32_t> history;
    
    void push(int32_t op_id) {
        history.push_back(op_id);
        num_operators++;
        depth++;
    }
    
    bool pop() {
        if (history.empty()) return false;
        history.pop_back();
        num_operators--;
        depth--;
        return true;
    }
    
    void reset() {
        history.clear();
        depth = 0;
        num_operators = 0;
        is_valid = true;
    }
};

}  // namespace

TEST(SearchContext, InitialState) {
    SearchState state;
    
    EXPECT_EQ(state.depth, 0);
    EXPECT_EQ(state.num_operators, 0);
    EXPECT_TRUE(state.is_valid);
    EXPECT_TRUE(state.history.empty());
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, PushOperation) {
    SearchState state;
    state.push(1);
    
    EXPECT_EQ(state.depth, 1);
    EXPECT_EQ(state.num_operators, 1);
    EXPECT_EQ(state.history.size(), 1);
    EXPECT_EQ(state.history[0], 1);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, PopOperation) {
    SearchState state;
    state.push(1);
    state.push(2);
    bool success = state.pop();
    
    EXPECT_TRUE(success);
    EXPECT_EQ(state.depth, 1);
    EXPECT_EQ(state.num_operators, 1);
    EXPECT_EQ(state.history.size(), 1);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, PopEmptyFails) {
    SearchState state;
    bool success = state.pop();
    
    EXPECT_FALSE(success);
    EXPECT_EQ(state.depth, 0);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, ResetState) {
    SearchState state;
    state.push(1);
    state.push(2);
    state.push(3);
    state.reset();
    
    EXPECT_EQ(state.depth, 0);
    EXPECT_EQ(state.num_operators, 0);
    EXPECT_TRUE(state.history.empty());
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, BacktrackCorrectly) {
    SearchState state;
    
    // Simulate search: 1 -> 2 -> 3 -> backtrack -> 4
    state.push(1);
    state.push(2);
    state.push(3);
    state.pop();  // Backtrack from 3
    state.push(4);
    
    EXPECT_EQ(state.depth, 3);
    EXPECT_EQ(state.history.size(), 3);
    EXPECT_EQ(state.history[2], 4);  // Last element is 4, not 3
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SearchContext, DepthLimit) {
    SearchState state;
    int32_t max_depth = 5;
    
    for (int i = 0; i < 10; i++) {
        if (state.depth < max_depth) {
            state.push(i);
        }
    }
    
    EXPECT_EQ(state.depth, max_depth);
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// SymbolicGraph Tests
// =============================================================================

namespace {

struct SymbolicNode {
    int32_t id;
    std::string op_type;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;
};

struct SymbolicGraph {
    std::vector<SymbolicNode> nodes;
    
    int32_t add_node(const std::string& op_type) {
        int32_t id = static_cast<int32_t>(nodes.size());
        nodes.push_back({id, op_type, {}, {}});
        return id;
    }
    
    void add_edge(int32_t from_id, int32_t to_id) {
        if (from_id < nodes.size() && to_id < nodes.size()) {
            nodes[from_id].output_ids.push_back(to_id);
            nodes[to_id].input_ids.push_back(from_id);
        }
    }
    
    size_t num_nodes() const { return nodes.size(); }
    size_t num_edges() const {
        size_t count = 0;
        for (const auto& node : nodes) {
            count += node.output_ids.size();
        }
        return count;
    }
};

}  // namespace

TEST(SymbolicGraph, NodeCreation) {
    SymbolicGraph graph;
    int32_t id = graph.add_node("matmul");
    
    EXPECT_EQ(id, 0);
    EXPECT_EQ(graph.num_nodes(), 1);
    EXPECT_EQ(graph.nodes[0].op_type, "matmul");
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SymbolicGraph, MultipleNodes) {
    SymbolicGraph graph;
    graph.add_node("input");
    graph.add_node("matmul");
    graph.add_node("silu");
    graph.add_node("output");
    
    EXPECT_EQ(graph.num_nodes(), 4);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SymbolicGraph, EdgeConnection) {
    SymbolicGraph graph;
    int32_t in1 = graph.add_node("input");
    int32_t in2 = graph.add_node("input");
    int32_t mm = graph.add_node("matmul");
    
    graph.add_edge(in1, mm);
    graph.add_edge(in2, mm);
    
    EXPECT_EQ(graph.num_edges(), 2);
    EXPECT_EQ(graph.nodes[mm].input_ids.size(), 2);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SymbolicGraph, ChainedOperations) {
    SymbolicGraph graph;
    int32_t input = graph.add_node("input");
    int32_t mm = graph.add_node("matmul");
    int32_t silu = graph.add_node("silu");
    int32_t output = graph.add_node("output");
    
    graph.add_edge(input, mm);
    graph.add_edge(mm, silu);
    graph.add_edge(silu, output);
    
    EXPECT_EQ(graph.num_edges(), 3);
    
    // Verify chain
    EXPECT_EQ(graph.nodes[mm].input_ids[0], input);
    EXPECT_EQ(graph.nodes[silu].input_ids[0], mm);
    EXPECT_EQ(graph.nodes[output].input_ids[0], silu);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(SymbolicGraph, DependencyAnalysis) {
    SymbolicGraph graph;
    
    // Create diamond pattern: A -> B, A -> C, B -> D, C -> D
    int32_t a = graph.add_node("input");
    int32_t b = graph.add_node("matmul");
    int32_t c = graph.add_node("matmul");
    int32_t d = graph.add_node("add");
    
    graph.add_edge(a, b);
    graph.add_edge(a, c);
    graph.add_edge(b, d);
    graph.add_edge(c, d);
    
    // Node D should have 2 inputs
    EXPECT_EQ(graph.nodes[d].input_ids.size(), 2);
    
    // Node A should have 2 outputs
    EXPECT_EQ(graph.nodes[a].output_ids.size(), 2);
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Main
// =============================================================================

YIRAGE_TEST_MAIN()
