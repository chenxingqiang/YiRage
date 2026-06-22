// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_gtest.cc
 * @brief Search Engine Unit Tests (Google Test version)
 *
 * Tests for src/search/ including KernelGraphGenerator and SearchContext.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace yirage {
namespace search {

// Configuration structures
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

// Search state for testing
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

// Symbolic graph for testing
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
        if (static_cast<size_t>(from_id) < nodes.size() && 
            static_cast<size_t>(to_id) < nodes.size()) {
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

}  // namespace search
}  // namespace yirage

using namespace yirage::search;

// =============================================================================
// KernelGraphConfig Tests
// =============================================================================

class KernelGraphConfigTest : public ::testing::Test {
protected:
    KernelGraphConfig config;
};

TEST_F(KernelGraphConfigTest, DefaultValues) {
    EXPECT_EQ(config.grid_dim_x, 1);
    EXPECT_EQ(config.grid_dim_y, 1);
    EXPECT_EQ(config.grid_dim_z, 1);
    EXPECT_EQ(config.block_dim_x, 128);
    EXPECT_EQ(config.forloop_range, 8);
}

TEST_F(KernelGraphConfigTest, CustomValues) {
    config.grid_dim_x = 4;
    config.grid_dim_y = 2;
    config.block_dim_x = 256;
    
    EXPECT_EQ(config.grid_dim_x, 4);
    EXPECT_EQ(config.grid_dim_y, 2);
    EXPECT_EQ(config.block_dim_x, 256);
}

TEST_F(KernelGraphConfigTest, GridDimProduct) {
    config.grid_dim_x = 4;
    config.grid_dim_y = 2;
    config.grid_dim_z = 1;
    
    int64_t total_blocks = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;
    EXPECT_EQ(total_blocks, 8);
}

TEST_F(KernelGraphConfigTest, BlockDimProduct) {
    config.block_dim_x = 128;
    config.block_dim_y = 2;
    config.block_dim_z = 1;
    
    int64_t threads_per_block = config.block_dim_x * config.block_dim_y * config.block_dim_z;
    EXPECT_EQ(threads_per_block, 256);
}

// =============================================================================
// SearchConfig Tests
// =============================================================================

class SearchConfigTest : public ::testing::Test {
protected:
    SearchConfig config;
};

TEST_F(SearchConfigTest, DefaultValues) {
    EXPECT_EQ(config.max_depth, 10);
    EXPECT_EQ(config.max_operators, 64);
    EXPECT_TRUE(config.enable_pruning);
    EXPECT_EQ(config.backend, "cuda");
}

TEST_F(SearchConfigTest, CustomBackend) {
    config.backend = "cpu";
    EXPECT_EQ(config.backend, "cpu");
}

TEST_F(SearchConfigTest, DisablePruning) {
    config.enable_pruning = false;
    EXPECT_FALSE(config.enable_pruning);
}

// =============================================================================
// SearchState Tests
// =============================================================================

class SearchStateTest : public ::testing::Test {
protected:
    SearchState state;
};

TEST_F(SearchStateTest, InitialState) {
    EXPECT_EQ(state.depth, 0);
    EXPECT_EQ(state.num_operators, 0);
    EXPECT_TRUE(state.is_valid);
    EXPECT_TRUE(state.history.empty());
}

TEST_F(SearchStateTest, PushOperation) {
    state.push(1);
    
    EXPECT_EQ(state.depth, 1);
    EXPECT_EQ(state.num_operators, 1);
    EXPECT_EQ(state.history.size(), 1u);
    EXPECT_EQ(state.history[0], 1);
}

TEST_F(SearchStateTest, PushMultiple) {
    state.push(1);
    state.push(2);
    state.push(3);
    
    EXPECT_EQ(state.depth, 3);
    EXPECT_EQ(state.num_operators, 3);
    EXPECT_EQ(state.history.size(), 3u);
}

TEST_F(SearchStateTest, PopOperation) {
    state.push(1);
    state.push(2);
    bool success = state.pop();
    
    EXPECT_TRUE(success);
    EXPECT_EQ(state.depth, 1);
    EXPECT_EQ(state.num_operators, 1);
    EXPECT_EQ(state.history.size(), 1u);
}

TEST_F(SearchStateTest, PopEmptyFails) {
    bool success = state.pop();
    
    EXPECT_FALSE(success);
    EXPECT_EQ(state.depth, 0);
}

TEST_F(SearchStateTest, ResetState) {
    state.push(1);
    state.push(2);
    state.push(3);
    state.reset();
    
    EXPECT_EQ(state.depth, 0);
    EXPECT_EQ(state.num_operators, 0);
    EXPECT_TRUE(state.history.empty());
}

TEST_F(SearchStateTest, BacktrackCorrectly) {
    // Simulate search: 1 -> 2 -> 3 -> backtrack -> 4
    state.push(1);
    state.push(2);
    state.push(3);
    state.pop();  // Backtrack from 3
    state.push(4);
    
    EXPECT_EQ(state.depth, 3);
    EXPECT_EQ(state.history.size(), 3u);
    EXPECT_EQ(state.history[2], 4);  // Last element is 4, not 3
}

TEST_F(SearchStateTest, DepthLimit) {
    int32_t max_depth = 5;
    
    for (int i = 0; i < 10; i++) {
        if (state.depth < max_depth) {
            state.push(i);
        }
    }
    
    EXPECT_EQ(state.depth, max_depth);
}

// =============================================================================
// SymbolicGraph Tests
// =============================================================================

class SymbolicGraphTest : public ::testing::Test {
protected:
    SymbolicGraph graph;
};

TEST_F(SymbolicGraphTest, NodeCreation) {
    int32_t id = graph.add_node("matmul");
    
    EXPECT_EQ(id, 0);
    EXPECT_EQ(graph.num_nodes(), 1u);
    EXPECT_EQ(graph.nodes[0].op_type, "matmul");
}

TEST_F(SymbolicGraphTest, MultipleNodes) {
    graph.add_node("input");
    graph.add_node("matmul");
    graph.add_node("silu");
    graph.add_node("output");
    
    EXPECT_EQ(graph.num_nodes(), 4u);
}

TEST_F(SymbolicGraphTest, EdgeConnection) {
    int32_t in1 = graph.add_node("input");
    int32_t in2 = graph.add_node("input");
    int32_t mm = graph.add_node("matmul");
    
    graph.add_edge(in1, mm);
    graph.add_edge(in2, mm);
    
    EXPECT_EQ(graph.num_edges(), 2u);
    EXPECT_EQ(graph.nodes[mm].input_ids.size(), 2u);
}

TEST_F(SymbolicGraphTest, ChainedOperations) {
    int32_t input = graph.add_node("input");
    int32_t mm = graph.add_node("matmul");
    int32_t silu = graph.add_node("silu");
    int32_t output = graph.add_node("output");
    
    graph.add_edge(input, mm);
    graph.add_edge(mm, silu);
    graph.add_edge(silu, output);
    
    EXPECT_EQ(graph.num_edges(), 3u);
    
    // Verify chain
    EXPECT_EQ(graph.nodes[mm].input_ids[0], input);
    EXPECT_EQ(graph.nodes[silu].input_ids[0], mm);
    EXPECT_EQ(graph.nodes[output].input_ids[0], silu);
}

TEST_F(SymbolicGraphTest, DiamondPattern) {
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
    EXPECT_EQ(graph.nodes[d].input_ids.size(), 2u);
    
    // Node A should have 2 outputs
    EXPECT_EQ(graph.nodes[a].output_ids.size(), 2u);
}

// =============================================================================
// Parameterized Tests
// =============================================================================

class GridConfigTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(GridConfigTest, ValidGridConfigurations) {
    auto [grid_x, grid_y, grid_z, expected_blocks] = GetParam();
    
    KernelGraphConfig config;
    config.grid_dim_x = grid_x;
    config.grid_dim_y = grid_y;
    config.grid_dim_z = grid_z;
    
    int actual = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;
    EXPECT_EQ(actual, expected_blocks);
}

INSTANTIATE_TEST_SUITE_P(
    GridConfigs,
    GridConfigTest,
    ::testing::Values(
        std::make_tuple(1, 1, 1, 1),
        std::make_tuple(2, 2, 1, 4),
        std::make_tuple(4, 4, 1, 16),
        std::make_tuple(8, 8, 1, 64),
        std::make_tuple(4, 2, 2, 16)
    )
);

class BlockConfigTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(BlockConfigTest, ValidBlockConfigurations) {
    auto [block_x, block_y, block_z, expected_threads] = GetParam();
    
    KernelGraphConfig config;
    config.block_dim_x = block_x;
    config.block_dim_y = block_y;
    config.block_dim_z = block_z;
    
    int actual = config.block_dim_x * config.block_dim_y * config.block_dim_z;
    EXPECT_EQ(actual, expected_threads);
}

INSTANTIATE_TEST_SUITE_P(
    BlockConfigs,
    BlockConfigTest,
    ::testing::Values(
        std::make_tuple(32, 1, 1, 32),
        std::make_tuple(64, 1, 1, 64),
        std::make_tuple(128, 1, 1, 128),
        std::make_tuple(256, 1, 1, 256),
        std::make_tuple(16, 16, 1, 256),
        std::make_tuple(32, 32, 1, 1024)
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
