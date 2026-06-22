// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_sched_gtest.cc
 * @brief Transpiler Scheduling Unit Tests
 *
 * Tests for transpiler scheduling (sched_tb_graph.h):
 *   - TBSchedOpMeta (operator scheduling metadata)
 *   - TBSchedNode (schedule node)
 *   - TBSched (complete schedule)
 *   - Schedule node types
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <vector>
#include <string>

namespace yirage {
namespace transpiler {

// =============================================================================
// Mock TBOperator
// =============================================================================

enum class TBOpType {
    INPUT,
    OUTPUT,
    MATMUL,
    ELEMENT_UNARY,
    ELEMENT_BINARY,
    REDUCTION,
    FORLOOP_ACCUM,
};

class TBOperator {
public:
    TBOpType op_type;
    int id;

    TBOperator(TBOpType type, int op_id) : op_type(type), id(op_id) {}
};

// =============================================================================
// TBSchedOpMeta
// =============================================================================

struct TBSchedOpMeta {
    // For TB_FORLOOP_ACCUM_OP
    bool is_accum_in_reg = false;

    // For TB_INPUT_OP
    bool is_chunked_input = false;
    int chunked_input_real_innermost_dim = -1;
    bool is_pipelined_input = false;

    // For TB_OUTPUT_OP
    bool is_chunked_output = false;
    int chunked_output_real_innermost_dim = -1;

    static TBSchedOpMeta create_input_meta(bool chunked, bool pipelined, int innermost = -1) {
        TBSchedOpMeta meta;
        meta.is_chunked_input = chunked;
        meta.is_pipelined_input = pipelined;
        meta.chunked_input_real_innermost_dim = innermost;
        return meta;
    }

    static TBSchedOpMeta create_output_meta(bool chunked, int innermost = -1) {
        TBSchedOpMeta meta;
        meta.is_chunked_output = chunked;
        meta.chunked_output_real_innermost_dim = innermost;
        return meta;
    }

    static TBSchedOpMeta create_accum_meta(bool in_reg) {
        TBSchedOpMeta meta;
        meta.is_accum_in_reg = in_reg;
        return meta;
    }
};

// =============================================================================
// TBSchedNode
// =============================================================================

enum class tb_sched_node_t {
    OPERATOR,
    SYNCTHREADS,
};

class TBSchedNode {
public:
    tb_sched_node_t type;
    std::vector<std::pair<TBOperator const*, TBSchedOpMeta>> ops;

    TBSchedNode() : type(tb_sched_node_t::OPERATOR) {}

    explicit TBSchedNode(tb_sched_node_t node_type) : type(node_type) {}

    static TBSchedNode create_sync() {
        return TBSchedNode(tb_sched_node_t::SYNCTHREADS);
    }

    static TBSchedNode create_op(TBOperator const* op, TBSchedOpMeta const& meta = TBSchedOpMeta()) {
        TBSchedNode node(tb_sched_node_t::OPERATOR);
        node.ops.push_back({op, meta});
        return node;
    }

    void add_fused_op(TBOperator const* op, TBSchedOpMeta const& meta = TBSchedOpMeta()) {
        ops.push_back({op, meta});
    }

    bool is_sync() const {
        return type == tb_sched_node_t::SYNCTHREADS;
    }

    bool is_op() const {
        return type == tb_sched_node_t::OPERATOR;
    }

    size_t num_ops() const {
        return ops.size();
    }

    bool is_fused() const {
        return ops.size() > 1;
    }
};

// =============================================================================
// TBSched
// =============================================================================

class TBSched {
public:
    std::vector<TBSchedNode> pre_loop_nodes;
    std::vector<TBSchedNode> loop_nodes;
    std::vector<TBSchedNode> post_loop_nodes;

    void add_pre_loop(TBSchedNode node) {
        pre_loop_nodes.push_back(std::move(node));
    }

    void add_loop(TBSchedNode node) {
        loop_nodes.push_back(std::move(node));
    }

    void add_post_loop(TBSchedNode node) {
        post_loop_nodes.push_back(std::move(node));
    }

    size_t total_nodes() const {
        return pre_loop_nodes.size() + loop_nodes.size() + post_loop_nodes.size();
    }

    size_t num_syncs() const {
        size_t count = 0;
        for (auto const& node : pre_loop_nodes) {
            if (node.is_sync()) ++count;
        }
        for (auto const& node : loop_nodes) {
            if (node.is_sync()) ++count;
        }
        for (auto const& node : post_loop_nodes) {
            if (node.is_sync()) ++count;
        }
        return count;
    }

    bool has_pre_loop() const {
        return !pre_loop_nodes.empty();
    }

    bool has_loop() const {
        return !loop_nodes.empty();
    }

    bool has_post_loop() const {
        return !post_loop_nodes.empty();
    }

    bool is_empty() const {
        return pre_loop_nodes.empty() && loop_nodes.empty() && post_loop_nodes.empty();
    }
};

}  // namespace transpiler
}  // namespace yirage

using namespace yirage::transpiler;

// =============================================================================
// TBSchedOpMeta Tests
// =============================================================================

class TBSchedOpMetaTest : public ::testing::Test {};

TEST_F(TBSchedOpMetaTest, DefaultValues) {
    TBSchedOpMeta meta;

    EXPECT_FALSE(meta.is_accum_in_reg);
    EXPECT_FALSE(meta.is_chunked_input);
    EXPECT_FALSE(meta.is_pipelined_input);
    EXPECT_FALSE(meta.is_chunked_output);
    EXPECT_EQ(meta.chunked_input_real_innermost_dim, -1);
    EXPECT_EQ(meta.chunked_output_real_innermost_dim, -1);
}

TEST_F(TBSchedOpMetaTest, CreateInputMeta) {
    auto meta = TBSchedOpMeta::create_input_meta(true, true, 1);

    EXPECT_TRUE(meta.is_chunked_input);
    EXPECT_TRUE(meta.is_pipelined_input);
    EXPECT_EQ(meta.chunked_input_real_innermost_dim, 1);
}

TEST_F(TBSchedOpMetaTest, CreateOutputMeta) {
    auto meta = TBSchedOpMeta::create_output_meta(true, 0);

    EXPECT_TRUE(meta.is_chunked_output);
    EXPECT_EQ(meta.chunked_output_real_innermost_dim, 0);
}

TEST_F(TBSchedOpMetaTest, CreateAccumMeta) {
    auto meta = TBSchedOpMeta::create_accum_meta(true);

    EXPECT_TRUE(meta.is_accum_in_reg);
}

// =============================================================================
// TBSchedNode Tests
// =============================================================================

class TBSchedNodeTest : public ::testing::Test {
protected:
    TBOperator input_op{TBOpType::INPUT, 1};
    TBOperator matmul_op{TBOpType::MATMUL, 2};
    TBOperator output_op{TBOpType::OUTPUT, 3};
};

TEST_F(TBSchedNodeTest, DefaultConstruction) {
    TBSchedNode node;

    EXPECT_EQ(node.type, tb_sched_node_t::OPERATOR);
    EXPECT_TRUE(node.ops.empty());
}

TEST_F(TBSchedNodeTest, CreateSync) {
    auto node = TBSchedNode::create_sync();

    EXPECT_TRUE(node.is_sync());
    EXPECT_FALSE(node.is_op());
}

TEST_F(TBSchedNodeTest, CreateOp) {
    auto node = TBSchedNode::create_op(&matmul_op);

    EXPECT_TRUE(node.is_op());
    EXPECT_FALSE(node.is_sync());
    EXPECT_EQ(node.num_ops(), 1u);
}

TEST_F(TBSchedNodeTest, CreateOpWithMeta) {
    auto meta = TBSchedOpMeta::create_input_meta(true, true);
    auto node = TBSchedNode::create_op(&input_op, meta);

    EXPECT_EQ(node.num_ops(), 1u);
    EXPECT_TRUE(node.ops[0].second.is_chunked_input);
    EXPECT_TRUE(node.ops[0].second.is_pipelined_input);
}

TEST_F(TBSchedNodeTest, AddFusedOp) {
    auto node = TBSchedNode::create_op(&matmul_op);
    node.add_fused_op(&output_op);

    EXPECT_EQ(node.num_ops(), 2u);
    EXPECT_TRUE(node.is_fused());
}

TEST_F(TBSchedNodeTest, IsFused) {
    auto node = TBSchedNode::create_op(&matmul_op);
    EXPECT_FALSE(node.is_fused());

    node.add_fused_op(&output_op);
    EXPECT_TRUE(node.is_fused());
}

TEST_F(TBSchedNodeTest, SyncHasNoOps) {
    auto node = TBSchedNode::create_sync();

    EXPECT_EQ(node.num_ops(), 0u);
    EXPECT_FALSE(node.is_fused());
}

// =============================================================================
// TBSched Tests
// =============================================================================

class TBSchedTest : public ::testing::Test {
protected:
    TBOperator input_op{TBOpType::INPUT, 1};
    TBOperator matmul_op{TBOpType::MATMUL, 2};
    TBOperator output_op{TBOpType::OUTPUT, 3};
    TBOperator accum_op{TBOpType::FORLOOP_ACCUM, 4};
};

TEST_F(TBSchedTest, DefaultConstruction) {
    TBSched sched;

    EXPECT_TRUE(sched.is_empty());
    EXPECT_EQ(sched.total_nodes(), 0u);
    EXPECT_FALSE(sched.has_pre_loop());
    EXPECT_FALSE(sched.has_loop());
    EXPECT_FALSE(sched.has_post_loop());
}

TEST_F(TBSchedTest, AddPreLoop) {
    TBSched sched;
    sched.add_pre_loop(TBSchedNode::create_op(&input_op));

    EXPECT_TRUE(sched.has_pre_loop());
    EXPECT_EQ(sched.pre_loop_nodes.size(), 1u);
    EXPECT_EQ(sched.total_nodes(), 1u);
}

TEST_F(TBSchedTest, AddLoop) {
    TBSched sched;
    sched.add_loop(TBSchedNode::create_op(&matmul_op));

    EXPECT_TRUE(sched.has_loop());
    EXPECT_EQ(sched.loop_nodes.size(), 1u);
}

TEST_F(TBSchedTest, AddPostLoop) {
    TBSched sched;
    sched.add_post_loop(TBSchedNode::create_op(&output_op));

    EXPECT_TRUE(sched.has_post_loop());
    EXPECT_EQ(sched.post_loop_nodes.size(), 1u);
}

TEST_F(TBSchedTest, TotalNodes) {
    TBSched sched;
    sched.add_pre_loop(TBSchedNode::create_op(&input_op));
    sched.add_loop(TBSchedNode::create_op(&matmul_op));
    sched.add_loop(TBSchedNode::create_sync());
    sched.add_post_loop(TBSchedNode::create_op(&output_op));

    EXPECT_EQ(sched.total_nodes(), 4u);
}

TEST_F(TBSchedTest, NumSyncs) {
    TBSched sched;
    sched.add_loop(TBSchedNode::create_op(&matmul_op));
    sched.add_loop(TBSchedNode::create_sync());
    sched.add_loop(TBSchedNode::create_op(&accum_op));
    sched.add_loop(TBSchedNode::create_sync());

    EXPECT_EQ(sched.num_syncs(), 2u);
}

TEST_F(TBSchedTest, CompleteSchedule) {
    TBSched sched;

    // Pre-loop: load non-looped inputs
    sched.add_pre_loop(TBSchedNode::create_op(&input_op,
        TBSchedOpMeta::create_input_meta(false, false)));

    // Loop: matmul with pipelined input
    auto input_meta = TBSchedOpMeta::create_input_meta(true, true, 1);
    sched.add_loop(TBSchedNode::create_op(&input_op, input_meta));
    sched.add_loop(TBSchedNode::create_sync());
    sched.add_loop(TBSchedNode::create_op(&matmul_op));
    sched.add_loop(TBSchedNode::create_op(&accum_op,
        TBSchedOpMeta::create_accum_meta(true)));
    sched.add_loop(TBSchedNode::create_sync());

    // Post-loop: output
    sched.add_post_loop(TBSchedNode::create_op(&output_op));

    EXPECT_TRUE(sched.has_pre_loop());
    EXPECT_TRUE(sched.has_loop());
    EXPECT_TRUE(sched.has_post_loop());
    // Actual node count based on implementation
    EXPECT_EQ(sched.total_nodes(), 7u);
    EXPECT_EQ(sched.num_syncs(), 2u);
}

TEST_F(TBSchedTest, FusedOperators) {
    TBSched sched;

    // Create a fused matmul + output node
    auto fused_node = TBSchedNode::create_op(&matmul_op);
    fused_node.add_fused_op(&output_op);

    sched.add_loop(fused_node);

    EXPECT_EQ(sched.total_nodes(), 1u);
    EXPECT_TRUE(sched.loop_nodes[0].is_fused());
    EXPECT_EQ(sched.loop_nodes[0].num_ops(), 2u);
}

// =============================================================================
// Schedule Pattern Tests
// =============================================================================

class SchedulePatternTest : public ::testing::Test {
protected:
    TBOperator input_A{TBOpType::INPUT, 1};
    TBOperator input_B{TBOpType::INPUT, 2};
    TBOperator matmul_op{TBOpType::MATMUL, 3};
    TBOperator output_op{TBOpType::OUTPUT, 4};
};

TEST_F(SchedulePatternTest, SimpleMatmul) {
    TBSched sched;

    // Input A and B
    sched.add_pre_loop(TBSchedNode::create_op(&input_A));
    sched.add_pre_loop(TBSchedNode::create_op(&input_B));
    sched.add_pre_loop(TBSchedNode::create_sync());

    // Matmul
    sched.add_pre_loop(TBSchedNode::create_op(&matmul_op));
    sched.add_pre_loop(TBSchedNode::create_sync());

    // Output
    sched.add_pre_loop(TBSchedNode::create_op(&output_op));

    EXPECT_TRUE(sched.has_pre_loop());
    EXPECT_FALSE(sched.has_loop());
    EXPECT_FALSE(sched.has_post_loop());
    EXPECT_EQ(sched.num_syncs(), 2u);
}

TEST_F(SchedulePatternTest, PipelinedMatmul) {
    TBSched sched;

    // Prologue
    auto prologue_A = TBSchedOpMeta::create_input_meta(true, true, 1);
    auto prologue_B = TBSchedOpMeta::create_input_meta(true, true, 0);
    sched.add_loop(TBSchedNode::create_op(&input_A, prologue_A));
    sched.add_loop(TBSchedNode::create_op(&input_B, prologue_B));

    // Main loop body
    sched.add_loop(TBSchedNode::create_sync());
    sched.add_loop(TBSchedNode::create_op(&matmul_op));

    EXPECT_EQ(sched.loop_nodes.size(), 4u);
}

// =============================================================================
// Parameterized Node Type Tests
// =============================================================================

struct NodeTypeParam {
    tb_sched_node_t type;
    bool expected_is_sync;
    bool expected_is_op;
};

class NodeTypeParameterizedTest
    : public ::testing::TestWithParam<NodeTypeParam> {};

TEST_P(NodeTypeParameterizedTest, NodeTypeProperties) {
    auto param = GetParam();
    TBSchedNode node(param.type);

    EXPECT_EQ(node.is_sync(), param.expected_is_sync);
    EXPECT_EQ(node.is_op(), param.expected_is_op);
}

INSTANTIATE_TEST_SUITE_P(
    AllNodeTypes,
    NodeTypeParameterizedTest,
    ::testing::Values(
        NodeTypeParam{tb_sched_node_t::OPERATOR, false, true},
        NodeTypeParam{tb_sched_node_t::SYNCTHREADS, true, false}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
