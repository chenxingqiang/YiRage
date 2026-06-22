// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_distributed_gtest.cc
 * @brief Distributed Search Module Unit Tests
 *
 * Tests for distributed search components:
 *   - SearchPartition creation and partitioning
 *   - PartitionStrategy
 *   - PartitionConfig
 *   - SearchFeedback
 *   - SearchCallback
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

namespace yirage {
namespace search {

// =============================================================================
// Vector Types
// =============================================================================

struct int3 {
    int x = -1, y = -1, z = -1;
};

struct dim3 {
    int x = 1, y = 1, z = 1;
    
    bool operator==(dim3 const& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// =============================================================================
// Mock GeneratorConfig
// =============================================================================

struct GeneratorConfig {
    std::vector<dim3> grid_dim_to_explore = {{1, 1, 1}, {2, 2, 1}, {4, 4, 1}, {8, 8, 1}};
    std::vector<dim3> block_dim_to_explore = {{128, 1, 1}, {256, 1, 1}};
    std::vector<int3> imap_to_explore = {{0, 1, -1}, {1, 0, -1}};
    std::vector<int3> omap_to_explore = {{0, 1, -1}};
    std::vector<int> fmap_to_explore = {-1, 0, 1};
    std::vector<int> frange_to_explore = {1, 2, 4, 8};
};

// =============================================================================
// PartitionStrategy
// =============================================================================

enum class PartitionStrategy {
    BY_GRID_DIM,
    BY_BLOCK_DIM,
    BY_CONFIG_HASH,
    ROUND_ROBIN,
};

// =============================================================================
// PartitionConfig
// =============================================================================

struct PartitionConfig {
    int num_partitions = 1;
    PartitionStrategy strategy = PartitionStrategy::BY_GRID_DIM;
    bool balance_by_estimate = true;
};

// =============================================================================
// SearchPartition
// =============================================================================

struct SearchPartition {
    int partition_id = 0;
    int total_partitions = 1;
    
    std::vector<dim3> grid_dim_range;
    std::vector<dim3> block_dim_range;
    std::vector<int3> imap_range;
    std::vector<int3> omap_range;
    std::vector<int> fmap_range;
    std::vector<int> frange_range;
    
    size_t estimated_candidates = 0;
    
    static std::vector<SearchPartition> create_partitions(
            GeneratorConfig const& config, int num_partitions) {
        std::vector<SearchPartition> partitions;
        
        if (num_partitions <= 0) return partitions;
        
        // Partition by grid dimensions
        size_t num_grid_dims = config.grid_dim_to_explore.size();
        size_t per_partition = (num_grid_dims + num_partitions - 1) / num_partitions;
        
        for (int i = 0; i < num_partitions; ++i) {
            SearchPartition p;
            p.partition_id = i;
            p.total_partitions = num_partitions;
            
            // Assign grid dims to this partition
            size_t start = i * per_partition;
            size_t end = std::min(start + per_partition, num_grid_dims);
            
            for (size_t j = start; j < end; ++j) {
                p.grid_dim_range.push_back(config.grid_dim_to_explore[j]);
            }
            
            // Copy all other ranges
            p.block_dim_range = config.block_dim_to_explore;
            p.imap_range = config.imap_to_explore;
            p.omap_range = config.omap_to_explore;
            p.fmap_range = config.fmap_to_explore;
            p.frange_range = config.frange_to_explore;
            
            // Estimate candidates
            p.estimated_candidates = p.grid_dim_range.size() *
                                     p.block_dim_range.size() *
                                     p.frange_range.size();
            
            if (!p.grid_dim_range.empty()) {
                partitions.push_back(std::move(p));
            }
        }
        
        return partitions;
    }
    
    bool contains(dim3 grid_dim, dim3 block_dim) const {
        bool grid_match = false;
        for (auto const& g : grid_dim_range) {
            if (g == grid_dim) {
                grid_match = true;
                break;
            }
        }
        
        bool block_match = false;
        for (auto const& b : block_dim_range) {
            if (b == block_dim) {
                block_match = true;
                break;
            }
        }
        
        return grid_match && block_match;
    }
};

// =============================================================================
// SearchFeedback
// =============================================================================

struct SearchFeedback {
    int worker_id = 0;
    int partition_id = 0;
    
    // Progress
    int candidates_explored = 0;
    int candidates_valid = 0;
    int candidates_best = 0;
    
    // Performance
    float best_latency_ms = std::numeric_limits<float>::max();
    float elapsed_time_seconds = 0.0f;
    float throughput_candidates_per_sec = 0.0f;
    
    // Status
    enum class Status {
        RUNNING,
        COMPLETED,
        TIMEOUT,
        ERROR
    };
    Status status = Status::RUNNING;
    std::string error_message;
    
    void update_throughput() {
        if (elapsed_time_seconds > 0) {
            throughput_candidates_per_sec = candidates_explored / elapsed_time_seconds;
        }
    }
    
    bool is_complete() const {
        return status == Status::COMPLETED ||
               status == Status::TIMEOUT ||
               status == Status::ERROR;
    }
};

// =============================================================================
// SearchCallback
// =============================================================================

class SearchCallback {
public:
    using FeedbackHandler = std::function<void(SearchFeedback const&)>;
    using ResultHandler = std::function<void(int partition_id, void* result)>;
    
    void set_feedback_handler(FeedbackHandler handler) {
        feedback_handler_ = std::move(handler);
    }
    
    void set_result_handler(ResultHandler handler) {
        result_handler_ = std::move(handler);
    }
    
    void on_feedback(SearchFeedback const& feedback) {
        if (feedback_handler_) {
            feedback_handler_(feedback);
        }
        feedbacks_.push_back(feedback);
    }
    
    void on_result(int partition_id, void* result) {
        if (result_handler_) {
            result_handler_(partition_id, result);
        }
    }
    
    std::vector<SearchFeedback> const& get_all_feedbacks() const {
        return feedbacks_;
    }
    
    SearchFeedback aggregate_feedbacks() const {
        SearchFeedback agg;
        agg.status = SearchFeedback::Status::COMPLETED;
        
        for (auto const& f : feedbacks_) {
            agg.candidates_explored += f.candidates_explored;
            agg.candidates_valid += f.candidates_valid;
            agg.best_latency_ms = std::min(agg.best_latency_ms, f.best_latency_ms);
            
            if (f.status == SearchFeedback::Status::ERROR) {
                agg.status = SearchFeedback::Status::ERROR;
            }
        }
        
        return agg;
    }
    
private:
    FeedbackHandler feedback_handler_;
    ResultHandler result_handler_;
    std::vector<SearchFeedback> feedbacks_;
};

// =============================================================================
// PartitionedGenerator
// =============================================================================

class PartitionedGenerator {
public:
    PartitionedGenerator(GeneratorConfig const& config, PartitionConfig const& partition_config)
        : config_(config), partition_config_(partition_config) {
        partitions_ = SearchPartition::create_partitions(config, partition_config.num_partitions);
    }
    
    size_t num_partitions() const { return partitions_.size(); }
    
    SearchPartition const& get_partition(int index) const {
        return partitions_.at(index);
    }
    
    std::vector<SearchPartition> const& get_all_partitions() const {
        return partitions_;
    }
    
    size_t total_estimated_candidates() const {
        size_t total = 0;
        for (auto const& p : partitions_) {
            total += p.estimated_candidates;
        }
        return total;
    }
    
private:
    GeneratorConfig config_;
    PartitionConfig partition_config_;
    std::vector<SearchPartition> partitions_;
};

}  // namespace search
}  // namespace yirage

using namespace yirage::search;

// =============================================================================
// SearchPartition Tests
// =============================================================================

class SearchPartitionTest : public ::testing::Test {
protected:
    GeneratorConfig config;
};

TEST_F(SearchPartitionTest, CreateSinglePartition) {
    auto partitions = SearchPartition::create_partitions(config, 1);
    EXPECT_EQ(partitions.size(), 1u);
    EXPECT_EQ(partitions[0].grid_dim_range.size(), config.grid_dim_to_explore.size());
}

TEST_F(SearchPartitionTest, CreateMultiplePartitions) {
    auto partitions = SearchPartition::create_partitions(config, 2);
    EXPECT_GE(partitions.size(), 1u);
    EXPECT_LE(partitions.size(), 2u);
}

TEST_F(SearchPartitionTest, PartitionIds) {
    auto partitions = SearchPartition::create_partitions(config, 4);
    
    for (size_t i = 0; i < partitions.size(); ++i) {
        EXPECT_EQ(partitions[i].partition_id, static_cast<int>(i));
    }
}

TEST_F(SearchPartitionTest, TotalPartitionsField) {
    auto partitions = SearchPartition::create_partitions(config, 4);
    
    for (auto const& p : partitions) {
        EXPECT_EQ(p.total_partitions, 4);
    }
}

TEST_F(SearchPartitionTest, EstimatedCandidates) {
    auto partitions = SearchPartition::create_partitions(config, 1);
    EXPECT_GT(partitions[0].estimated_candidates, 0u);
}

TEST_F(SearchPartitionTest, Contains) {
    auto partitions = SearchPartition::create_partitions(config, 1);
    
    // Should contain a valid grid/block combo
    dim3 grid{1, 1, 1};
    dim3 block{128, 1, 1};
    EXPECT_TRUE(partitions[0].contains(grid, block));
    
    // Should not contain invalid combo
    dim3 invalid_grid{999, 999, 999};
    EXPECT_FALSE(partitions[0].contains(invalid_grid, block));
}

TEST_F(SearchPartitionTest, PartitionsCoverAllGridDims) {
    auto partitions = SearchPartition::create_partitions(config, 4);
    
    // Count total grid dims across all partitions
    size_t total_grid_dims = 0;
    for (auto const& p : partitions) {
        total_grid_dims += p.grid_dim_range.size();
    }
    
    EXPECT_EQ(total_grid_dims, config.grid_dim_to_explore.size());
}

// =============================================================================
// PartitionConfig Tests
// =============================================================================

class PartitionConfigTest : public ::testing::Test {};

TEST_F(PartitionConfigTest, DefaultValues) {
    PartitionConfig config;
    EXPECT_EQ(config.num_partitions, 1);
    EXPECT_EQ(config.strategy, PartitionStrategy::BY_GRID_DIM);
    EXPECT_TRUE(config.balance_by_estimate);
}

TEST_F(PartitionConfigTest, ConfigurePartitions) {
    PartitionConfig config;
    config.num_partitions = 8;
    config.strategy = PartitionStrategy::ROUND_ROBIN;
    config.balance_by_estimate = false;
    
    EXPECT_EQ(config.num_partitions, 8);
    EXPECT_EQ(config.strategy, PartitionStrategy::ROUND_ROBIN);
    EXPECT_FALSE(config.balance_by_estimate);
}

// =============================================================================
// SearchFeedback Tests
// =============================================================================

class SearchFeedbackTest : public ::testing::Test {};

TEST_F(SearchFeedbackTest, DefaultValues) {
    SearchFeedback feedback;
    EXPECT_EQ(feedback.worker_id, 0);
    EXPECT_EQ(feedback.candidates_explored, 0);
    EXPECT_EQ(feedback.status, SearchFeedback::Status::RUNNING);
}

TEST_F(SearchFeedbackTest, UpdateThroughput) {
    SearchFeedback feedback;
    feedback.candidates_explored = 1000;
    feedback.elapsed_time_seconds = 10.0f;
    
    feedback.update_throughput();
    EXPECT_FLOAT_EQ(feedback.throughput_candidates_per_sec, 100.0f);
}

TEST_F(SearchFeedbackTest, IsComplete) {
    SearchFeedback feedback;
    
    feedback.status = SearchFeedback::Status::RUNNING;
    EXPECT_FALSE(feedback.is_complete());
    
    feedback.status = SearchFeedback::Status::COMPLETED;
    EXPECT_TRUE(feedback.is_complete());
    
    feedback.status = SearchFeedback::Status::TIMEOUT;
    EXPECT_TRUE(feedback.is_complete());
    
    feedback.status = SearchFeedback::Status::ERROR;
    EXPECT_TRUE(feedback.is_complete());
}

TEST_F(SearchFeedbackTest, BestLatency) {
    SearchFeedback feedback;
    feedback.best_latency_ms = 1.5f;
    
    EXPECT_FLOAT_EQ(feedback.best_latency_ms, 1.5f);
}

// =============================================================================
// SearchCallback Tests
// =============================================================================

class SearchCallbackTest : public ::testing::Test {};

TEST_F(SearchCallbackTest, SetFeedbackHandler) {
    SearchCallback callback;
    int call_count = 0;
    
    callback.set_feedback_handler([&call_count](SearchFeedback const&) {
        ++call_count;
    });
    
    SearchFeedback feedback;
    callback.on_feedback(feedback);
    callback.on_feedback(feedback);
    
    EXPECT_EQ(call_count, 2);
}

TEST_F(SearchCallbackTest, SetResultHandler) {
    SearchCallback callback;
    int received_partition = -1;
    
    callback.set_result_handler([&received_partition](int partition_id, void*) {
        received_partition = partition_id;
    });
    
    callback.on_result(5, nullptr);
    EXPECT_EQ(received_partition, 5);
}

TEST_F(SearchCallbackTest, GetAllFeedbacks) {
    SearchCallback callback;
    
    SearchFeedback f1;
    f1.partition_id = 0;
    f1.candidates_explored = 100;
    
    SearchFeedback f2;
    f2.partition_id = 1;
    f2.candidates_explored = 200;
    
    callback.on_feedback(f1);
    callback.on_feedback(f2);
    
    auto const& feedbacks = callback.get_all_feedbacks();
    EXPECT_EQ(feedbacks.size(), 2u);
    EXPECT_EQ(feedbacks[0].candidates_explored, 100);
    EXPECT_EQ(feedbacks[1].candidates_explored, 200);
}

TEST_F(SearchCallbackTest, AggregateFeedbacks) {
    SearchCallback callback;
    
    SearchFeedback f1;
    f1.candidates_explored = 100;
    f1.candidates_valid = 10;
    f1.best_latency_ms = 2.0f;
    f1.status = SearchFeedback::Status::COMPLETED;
    
    SearchFeedback f2;
    f2.candidates_explored = 200;
    f2.candidates_valid = 20;
    f2.best_latency_ms = 1.5f;
    f2.status = SearchFeedback::Status::COMPLETED;
    
    callback.on_feedback(f1);
    callback.on_feedback(f2);
    
    auto agg = callback.aggregate_feedbacks();
    EXPECT_EQ(agg.candidates_explored, 300);
    EXPECT_EQ(agg.candidates_valid, 30);
    EXPECT_FLOAT_EQ(agg.best_latency_ms, 1.5f);  // Min of 2.0 and 1.5
}

TEST_F(SearchCallbackTest, AggregateWithError) {
    SearchCallback callback;
    
    SearchFeedback f1;
    f1.status = SearchFeedback::Status::COMPLETED;
    
    SearchFeedback f2;
    f2.status = SearchFeedback::Status::ERROR;
    
    callback.on_feedback(f1);
    callback.on_feedback(f2);
    
    auto agg = callback.aggregate_feedbacks();
    EXPECT_EQ(agg.status, SearchFeedback::Status::ERROR);
}

// =============================================================================
// PartitionedGenerator Tests
// =============================================================================

class PartitionedGeneratorTest : public ::testing::Test {
protected:
    GeneratorConfig gen_config;
};

TEST_F(PartitionedGeneratorTest, SinglePartition) {
    PartitionConfig part_config;
    part_config.num_partitions = 1;
    
    PartitionedGenerator generator(gen_config, part_config);
    EXPECT_EQ(generator.num_partitions(), 1u);
}

TEST_F(PartitionedGeneratorTest, MultiplePartitions) {
    PartitionConfig part_config;
    part_config.num_partitions = 4;
    
    PartitionedGenerator generator(gen_config, part_config);
    EXPECT_GE(generator.num_partitions(), 1u);
    EXPECT_LE(generator.num_partitions(), 4u);
}

TEST_F(PartitionedGeneratorTest, GetPartition) {
    PartitionConfig part_config;
    part_config.num_partitions = 2;
    
    PartitionedGenerator generator(gen_config, part_config);
    
    auto const& p0 = generator.get_partition(0);
    EXPECT_EQ(p0.partition_id, 0);
}

TEST_F(PartitionedGeneratorTest, GetAllPartitions) {
    PartitionConfig part_config;
    part_config.num_partitions = 4;
    
    PartitionedGenerator generator(gen_config, part_config);
    
    auto const& partitions = generator.get_all_partitions();
    EXPECT_EQ(partitions.size(), generator.num_partitions());
}

TEST_F(PartitionedGeneratorTest, TotalEstimatedCandidates) {
    PartitionConfig part_config;
    part_config.num_partitions = 4;
    
    PartitionedGenerator generator(gen_config, part_config);
    
    size_t total = generator.total_estimated_candidates();
    EXPECT_GT(total, 0u);
}

// =============================================================================
// Parameterized Partition Tests
// =============================================================================

struct PartitionTestParam {
    int num_partitions;
    size_t expected_min_partitions;
    size_t expected_max_partitions;
};

class PartitionParameterizedTest : public ::testing::TestWithParam<PartitionTestParam> {
protected:
    GeneratorConfig config;
};

TEST_P(PartitionParameterizedTest, PartitionCount) {
    auto param = GetParam();
    
    auto partitions = SearchPartition::create_partitions(config, param.num_partitions);
    
    EXPECT_GE(partitions.size(), param.expected_min_partitions);
    EXPECT_LE(partitions.size(), param.expected_max_partitions);
}

INSTANTIATE_TEST_SUITE_P(
    PartitionCounts,
    PartitionParameterizedTest,
    ::testing::Values(
        PartitionTestParam{1, 1, 1},
        PartitionTestParam{2, 1, 2},
        PartitionTestParam{4, 1, 4},
        PartitionTestParam{8, 1, 4},   // More partitions than grid dims
        PartitionTestParam{0, 0, 0}    // Edge case
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
