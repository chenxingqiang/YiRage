// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_config_gtest.cc
 * @brief Search Configuration and Strategy Module Unit Tests
 *
 * Tests for search configuration and strategy:
 *   - GeneratorConfig construction and defaults
 *   - SearchConfig parameters
 *   - CandidateConfig structure
 *   - SearchStrategy interface
 *   - SearchStrategyFactory
 *   - TBGraphConfig
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace yirage {

// Forward declarations
namespace type {

enum class BackendType {
    BT_CUDA = 0,
    BT_CUDNN = 1,
    BT_CUTLASS = 2,
    BT_ROCM = 3,
    BT_CPU = 4,
    BT_MKL = 5,
    BT_MPS = 6,
    BT_ASCEND = 7,
    BT_MACA = 8,
    BT_XPU = 9,
    BT_TPU = 10,
    BT_NKI = 11,
    BT_TRITON = 12,
    BT_FPGA = 13,
    BT_UNKNOWN = 14,
};

enum class KNOperatorType {
    KN_INPUT = 0,
    KN_OUTPUT = 1,
    KN_MATMUL = 2,
    KN_CONV2D = 3,
    KN_REDUCTION = 4,
    KN_CUSTOMIZED = 5,
};

enum class TBOperatorType {
    TB_INPUT = 0,
    TB_OUTPUT = 1,
    TB_MATMUL = 2,
    TB_EXP = 3,
    TB_SILU = 4,
    TB_MUL = 5,
    TB_ADD = 6,
    TB_DIV = 7,
    TB_REDUCTION = 8,
};

}  // namespace type

namespace kernel {

struct KernelConfig {
    int block_size = 256;
    int grid_size = 1;
    size_t shared_memory = 0;
    
    virtual ~KernelConfig() = default;
};

struct KernelMetrics {
    float latency_ms = 0.0f;
    float throughput_gflops = 0.0f;
    float memory_bandwidth_gbps = 0.0f;
    float occupancy = 0.0f;
};

// Mock graph
struct Graph {
    int num_inputs = 0;
    int num_outputs = 0;
    int num_ops = 0;
};

}  // namespace kernel

namespace search {

// =============================================================================
// Vector3 Types
// =============================================================================

struct int3 {
    int x = -1, y = -1, z = -1;
    
    bool operator==(int3 const& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct dim3 {
    int x = 1, y = 1, z = 1;
    
    bool operator==(dim3 const& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// =============================================================================
// VerifierType
// =============================================================================

enum class VerifierType {
    PROBABILISTIC_VERIFIER,
    FORMAL_VERIFIER,
};

// =============================================================================
// GeneratorConfig
// =============================================================================

struct GeneratorConfig {
    size_t max_num_threadblock_graph_op = 8;
    size_t max_num_kernel_graph_op = 16;
    size_t max_num_threadblock_graphs = 1;
    size_t max_num_threadblock_graph_inputs = 4;
    size_t max_num_threadblock_graph_outputs = 1;
    size_t search_thread = 1;
    
    VerifierType verifier_type = VerifierType::PROBABILISTIC_VERIFIER;
    type::BackendType backend_type = type::BackendType::BT_CUDA;
    int warp_size = 32;
    
    std::vector<type::KNOperatorType> knop_to_explore;
    std::vector<type::TBOperatorType> tbop_to_explore;
    std::vector<int3> imap_to_explore;
    std::vector<int3> omap_to_explore;
    std::vector<dim3> grid_dim_to_explore;
    std::vector<dim3> block_dim_to_explore;
    std::vector<int> fmap_to_explore;
    std::vector<int> frange_to_explore;
    int reduction_dimx = 1;
    
    bool randomized_branches = false;
    bool _enable_attention_specific_optimization = false;
    bool _enable_concat_matmul_transformation = false;
    
    void enable_attention_specific_optimization() {
        _enable_attention_specific_optimization = true;
    }
    
    void enable_concat_matmul_transformation() {
        _enable_concat_matmul_transformation = true;
    }
    
    static GeneratorConfig get_default_config() {
        GeneratorConfig config;
        config.max_num_threadblock_graph_op = 8;
        config.max_num_kernel_graph_op = 16;
        config.search_thread = 4;
        
        config.knop_to_explore = {
            type::KNOperatorType::KN_MATMUL,
            type::KNOperatorType::KN_REDUCTION,
        };
        
        config.tbop_to_explore = {
            type::TBOperatorType::TB_MATMUL,
            type::TBOperatorType::TB_EXP,
            type::TBOperatorType::TB_SILU,
            type::TBOperatorType::TB_MUL,
            type::TBOperatorType::TB_ADD,
            type::TBOperatorType::TB_REDUCTION,
        };
        
        config.grid_dim_to_explore = {{1, 1, 1}, {2, 2, 1}, {4, 4, 1}};
        config.block_dim_to_explore = {{128, 1, 1}, {256, 1, 1}};
        config.frange_to_explore = {1, 2, 4, 8};
        
        return config;
    }
};

// =============================================================================
// TBGraphConfig
// =============================================================================

struct TBGraphConfig {
    dim3 grid_dim = {1, 1, 1};
    dim3 block_dim = {128, 1, 1};
    std::vector<int3> imaps;
    std::vector<int> fmaps;
    int frange = 1;
    
    bool operator==(TBGraphConfig const& other) const {
        return grid_dim == other.grid_dim &&
               block_dim == other.block_dim &&
               imaps == other.imaps &&
               fmaps == other.fmaps &&
               frange == other.frange;
    }
    
    size_t hash() const {
        size_t h = 0;
        h ^= std::hash<int>()(grid_dim.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(block_dim.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(frange) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// =============================================================================
// SearchConfig
// =============================================================================

struct SearchConfig {
    int max_iterations = 1000;
    float timeout_seconds = 600.0f;
    bool use_cache = true;
    int random_seed = 42;
    
    enum class Strategy {
        GREEDY,
        BEAM,
        GENETIC,
        REINFORCEMENT
    };
    Strategy strategy = Strategy::GREEDY;
    
    int beam_width = 10;
    int population_size = 50;
    float mutation_rate = 0.1f;
    float crossover_rate = 0.7f;
    int num_warmup_iterations = 5;
    int num_profile_iterations = 10;
    
    virtual ~SearchConfig() = default;
};

// =============================================================================
// CandidateConfig
// =============================================================================

struct CandidateConfig {
    std::unique_ptr<kernel::KernelConfig> config;
    float score = 0.0f;
    kernel::KernelMetrics metrics;
    
    CandidateConfig() = default;
    
    CandidateConfig(std::unique_ptr<kernel::KernelConfig> cfg, float s = 0.0f)
        : config(std::move(cfg)), score(s) {}
};

// =============================================================================
// SearchStrategy Interface
// =============================================================================

class SearchStrategy {
public:
    virtual ~SearchStrategy() = default;
    
    virtual bool initialize(SearchConfig const& config) = 0;
    virtual std::vector<CandidateConfig> generate_candidates(kernel::Graph const& graph) = 0;
    virtual float evaluate_candidate(CandidateConfig& candidate, kernel::Graph const& graph) = 0;
    virtual kernel::KernelConfig* select_best_config(std::vector<CandidateConfig>& candidates) = 0;
    virtual std::unique_ptr<kernel::KernelConfig> optimize(kernel::Graph const& graph) = 0;
    virtual type::BackendType get_backend_type() const = 0;
    virtual std::string get_statistics() const = 0;
    
protected:
    SearchConfig config_;
    int num_candidates_generated_ = 0;
    int num_candidates_evaluated_ = 0;
    float best_score_ = 0.0f;
};

// =============================================================================
// Mock SearchStrategy Implementation
// =============================================================================

class MockSearchStrategy : public SearchStrategy {
public:
    explicit MockSearchStrategy(type::BackendType backend)
        : backend_type(backend) {}
    
    bool initialize(SearchConfig const& config) override {
        config_ = config;
        return true;
    }
    
    std::vector<CandidateConfig> generate_candidates(kernel::Graph const& graph) override {
        std::vector<CandidateConfig> candidates;
        
        for (int block_size : {128, 256, 512}) {
            auto config = std::make_unique<kernel::KernelConfig>();
            config->block_size = block_size;
            candidates.emplace_back(std::move(config), 0.0f);
            ++num_candidates_generated_;
        }
        
        return candidates;
    }
    
    float evaluate_candidate(CandidateConfig& candidate, kernel::Graph const& graph) override {
        ++num_candidates_evaluated_;
        
        // Simulate evaluation
        float score = 1.0f / candidate.config->block_size;
        candidate.score = score;
        candidate.metrics.latency_ms = 1.0f / score;
        
        if (score > best_score_) {
            best_score_ = score;
        }
        
        return score;
    }
    
    kernel::KernelConfig* select_best_config(std::vector<CandidateConfig>& candidates) override {
        kernel::KernelConfig* best = nullptr;
        float best_score = -1e9f;
        
        for (auto& c : candidates) {
            if (c.score > best_score) {
                best_score = c.score;
                best = c.config.get();
            }
        }
        
        return best;
    }
    
    std::unique_ptr<kernel::KernelConfig> optimize(kernel::Graph const& graph) override {
        auto candidates = generate_candidates(graph);
        
        for (auto& c : candidates) {
            evaluate_candidate(c, graph);
        }
        
        kernel::KernelConfig* best = select_best_config(candidates);
        
        auto result = std::make_unique<kernel::KernelConfig>();
        if (best) {
            *result = *best;
        }
        
        return result;
    }
    
    type::BackendType get_backend_type() const override {
        return backend_type;
    }
    
    std::string get_statistics() const override {
        return "Generated: " + std::to_string(num_candidates_generated_) +
               ", Evaluated: " + std::to_string(num_candidates_evaluated_) +
               ", Best: " + std::to_string(best_score_);
    }
    
private:
    type::BackendType backend_type;
};

// =============================================================================
// SearchStrategyFactory
// =============================================================================

class SearchStrategyFactory {
public:
    static std::unique_ptr<SearchStrategy> create_strategy(
            type::BackendType backend, SearchConfig const& config) {
        // Apply fallback mapping
        type::BackendType effective = get_effective_backend(backend);
        
        auto strategy = std::make_unique<MockSearchStrategy>(effective);
        strategy->initialize(config);
        return strategy;
    }
    
    static bool has_strategy(type::BackendType backend) {
        type::BackendType effective = get_effective_backend(backend);
        
        switch (effective) {
            case type::BackendType::BT_CUDA:
            case type::BackendType::BT_ROCM:
            case type::BackendType::BT_CPU:
            case type::BackendType::BT_MPS:
            case type::BackendType::BT_ASCEND:
                return true;
            default:
                return false;
        }
    }
    
    static type::BackendType get_effective_backend(type::BackendType backend) {
        switch (backend) {
            // CUDA software backends fall back to CUDA
            case type::BackendType::BT_CUDNN:
            case type::BackendType::BT_CUTLASS:
            case type::BackendType::BT_TRITON:
                return type::BackendType::BT_CUDA;
            
            // CPU software backends fall back to CPU
            case type::BackendType::BT_MKL:
                return type::BackendType::BT_CPU;
            
            // Hardware backends return themselves
            default:
                return backend;
        }
    }
    
    static std::vector<type::BackendType> get_supported_backends() {
        return {
            type::BackendType::BT_CUDA,
            type::BackendType::BT_CUDNN,
            type::BackendType::BT_CUTLASS,
            type::BackendType::BT_TRITON,
            type::BackendType::BT_ROCM,
            type::BackendType::BT_CPU,
            type::BackendType::BT_MKL,
            type::BackendType::BT_MPS,
            type::BackendType::BT_ASCEND,
        };
    }
};

}  // namespace search
}  // namespace yirage

using namespace yirage::search;
using namespace yirage::type;
using namespace yirage::kernel;

// =============================================================================
// GeneratorConfig Tests
// =============================================================================

class GeneratorConfigTest : public ::testing::Test {};

TEST_F(GeneratorConfigTest, DefaultConstruction) {
    GeneratorConfig config;
    EXPECT_EQ(config.max_num_threadblock_graph_op, 8u);
    EXPECT_EQ(config.max_num_kernel_graph_op, 16u);
    EXPECT_EQ(config.warp_size, 32);
    EXPECT_EQ(config.backend_type, BackendType::BT_CUDA);
}

TEST_F(GeneratorConfigTest, GetDefaultConfig) {
    auto config = GeneratorConfig::get_default_config();
    EXPECT_GT(config.search_thread, 0u);
    EXPECT_FALSE(config.knop_to_explore.empty());
    EXPECT_FALSE(config.tbop_to_explore.empty());
}

TEST_F(GeneratorConfigTest, EnableAttentionOptimization) {
    GeneratorConfig config;
    EXPECT_FALSE(config._enable_attention_specific_optimization);
    
    config.enable_attention_specific_optimization();
    EXPECT_TRUE(config._enable_attention_specific_optimization);
}

TEST_F(GeneratorConfigTest, EnableConcatMatmul) {
    GeneratorConfig config;
    EXPECT_FALSE(config._enable_concat_matmul_transformation);
    
    config.enable_concat_matmul_transformation();
    EXPECT_TRUE(config._enable_concat_matmul_transformation);
}

TEST_F(GeneratorConfigTest, ConfigureExploration) {
    GeneratorConfig config;
    config.knop_to_explore = {KNOperatorType::KN_MATMUL};
    config.grid_dim_to_explore = {{8, 8, 1}};
    config.frange_to_explore = {1, 2, 4};
    
    EXPECT_EQ(config.knop_to_explore.size(), 1u);
    EXPECT_EQ(config.grid_dim_to_explore.size(), 1u);
    EXPECT_EQ(config.frange_to_explore.size(), 3u);
}

// =============================================================================
// TBGraphConfig Tests
// =============================================================================

class TBGraphConfigTest : public ::testing::Test {};

TEST_F(TBGraphConfigTest, DefaultConstruction) {
    TBGraphConfig config;
    EXPECT_EQ(config.grid_dim.x, 1);
    EXPECT_EQ(config.block_dim.x, 128);
    EXPECT_EQ(config.frange, 1);
}

TEST_F(TBGraphConfigTest, Equality) {
    TBGraphConfig config1, config2;
    EXPECT_TRUE(config1 == config2);
    
    config2.frange = 2;
    EXPECT_FALSE(config1 == config2);
}

TEST_F(TBGraphConfigTest, Hash) {
    TBGraphConfig config1, config2;
    EXPECT_EQ(config1.hash(), config2.hash());
    
    config2.grid_dim.x = 4;
    EXPECT_NE(config1.hash(), config2.hash());
}

// =============================================================================
// SearchConfig Tests
// =============================================================================

class SearchConfigTest : public ::testing::Test {};

TEST_F(SearchConfigTest, DefaultConstruction) {
    SearchConfig config;
    EXPECT_EQ(config.max_iterations, 1000);
    EXPECT_FLOAT_EQ(config.timeout_seconds, 600.0f);
    EXPECT_TRUE(config.use_cache);
    EXPECT_EQ(config.strategy, SearchConfig::Strategy::GREEDY);
}

TEST_F(SearchConfigTest, BeamSearchParams) {
    SearchConfig config;
    config.strategy = SearchConfig::Strategy::BEAM;
    config.beam_width = 20;
    
    EXPECT_EQ(config.beam_width, 20);
}

TEST_F(SearchConfigTest, GeneticParams) {
    SearchConfig config;
    config.strategy = SearchConfig::Strategy::GENETIC;
    config.population_size = 100;
    config.mutation_rate = 0.2f;
    config.crossover_rate = 0.8f;
    
    EXPECT_EQ(config.population_size, 100);
    EXPECT_FLOAT_EQ(config.mutation_rate, 0.2f);
    EXPECT_FLOAT_EQ(config.crossover_rate, 0.8f);
}

// =============================================================================
// CandidateConfig Tests
// =============================================================================

class CandidateConfigTest : public ::testing::Test {};

TEST_F(CandidateConfigTest, DefaultConstruction) {
    CandidateConfig candidate;
    EXPECT_EQ(candidate.config, nullptr);
    EXPECT_FLOAT_EQ(candidate.score, 0.0f);
}

TEST_F(CandidateConfigTest, ConstructionWithConfig) {
    auto config = std::make_unique<KernelConfig>();
    config->block_size = 512;
    
    CandidateConfig candidate(std::move(config), 0.5f);
    
    EXPECT_NE(candidate.config, nullptr);
    EXPECT_EQ(candidate.config->block_size, 512);
    EXPECT_FLOAT_EQ(candidate.score, 0.5f);
}

TEST_F(CandidateConfigTest, Metrics) {
    CandidateConfig candidate;
    candidate.metrics.latency_ms = 1.5f;
    candidate.metrics.throughput_gflops = 100.0f;
    candidate.metrics.occupancy = 0.75f;
    
    EXPECT_FLOAT_EQ(candidate.metrics.latency_ms, 1.5f);
    EXPECT_FLOAT_EQ(candidate.metrics.throughput_gflops, 100.0f);
}

// =============================================================================
// SearchStrategy Tests
// =============================================================================

class SearchStrategyTest : public ::testing::Test {
protected:
    SearchConfig config;
    Graph graph{2, 1, 3};
};

TEST_F(SearchStrategyTest, Initialize) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    EXPECT_TRUE(strategy.initialize(config));
}

TEST_F(SearchStrategyTest, GenerateCandidates) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    strategy.initialize(config);
    
    auto candidates = strategy.generate_candidates(graph);
    EXPECT_FALSE(candidates.empty());
}

TEST_F(SearchStrategyTest, EvaluateCandidate) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    strategy.initialize(config);
    
    auto candidates = strategy.generate_candidates(graph);
    EXPECT_GT(candidates.size(), 0u);
    
    float score = strategy.evaluate_candidate(candidates[0], graph);
    EXPECT_GT(score, 0.0f);
}

TEST_F(SearchStrategyTest, SelectBestConfig) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    strategy.initialize(config);
    
    auto candidates = strategy.generate_candidates(graph);
    for (auto& c : candidates) {
        strategy.evaluate_candidate(c, graph);
    }
    
    auto best = strategy.select_best_config(candidates);
    EXPECT_NE(best, nullptr);
}

TEST_F(SearchStrategyTest, Optimize) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    strategy.initialize(config);
    
    auto result = strategy.optimize(graph);
    EXPECT_NE(result, nullptr);
}

TEST_F(SearchStrategyTest, GetStatistics) {
    MockSearchStrategy strategy(BackendType::BT_CUDA);
    strategy.initialize(config);
    strategy.optimize(graph);
    
    std::string stats = strategy.get_statistics();
    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Generated"), std::string::npos);
}

// =============================================================================
// SearchStrategyFactory Tests
// =============================================================================

class SearchStrategyFactoryTest : public ::testing::Test {};

TEST_F(SearchStrategyFactoryTest, CreateCUDAStrategy) {
    SearchConfig config;
    auto strategy = SearchStrategyFactory::create_strategy(BackendType::BT_CUDA, config);
    
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->get_backend_type(), BackendType::BT_CUDA);
}

TEST_F(SearchStrategyFactoryTest, CreateCPUStrategy) {
    SearchConfig config;
    auto strategy = SearchStrategyFactory::create_strategy(BackendType::BT_CPU, config);
    
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->get_backend_type(), BackendType::BT_CPU);
}

TEST_F(SearchStrategyFactoryTest, CUDNNFallbackToCUDA) {
    SearchConfig config;
    auto strategy = SearchStrategyFactory::create_strategy(BackendType::BT_CUDNN, config);
    
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->get_backend_type(), BackendType::BT_CUDA);
}

TEST_F(SearchStrategyFactoryTest, MKLFallbackToCPU) {
    SearchConfig config;
    auto strategy = SearchStrategyFactory::create_strategy(BackendType::BT_MKL, config);
    
    EXPECT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->get_backend_type(), BackendType::BT_CPU);
}

TEST_F(SearchStrategyFactoryTest, HasStrategyCUDA) {
    EXPECT_TRUE(SearchStrategyFactory::has_strategy(BackendType::BT_CUDA));
}

TEST_F(SearchStrategyFactoryTest, HasStrategyCUDNN) {
    EXPECT_TRUE(SearchStrategyFactory::has_strategy(BackendType::BT_CUDNN));  // Via fallback
}

TEST_F(SearchStrategyFactoryTest, GetEffectiveBackend) {
    EXPECT_EQ(SearchStrategyFactory::get_effective_backend(BackendType::BT_CUDA),
              BackendType::BT_CUDA);
    EXPECT_EQ(SearchStrategyFactory::get_effective_backend(BackendType::BT_CUDNN),
              BackendType::BT_CUDA);
    EXPECT_EQ(SearchStrategyFactory::get_effective_backend(BackendType::BT_TRITON),
              BackendType::BT_CUDA);
    EXPECT_EQ(SearchStrategyFactory::get_effective_backend(BackendType::BT_MKL),
              BackendType::BT_CPU);
}

TEST_F(SearchStrategyFactoryTest, GetSupportedBackends) {
    auto backends = SearchStrategyFactory::get_supported_backends();
    EXPECT_FALSE(backends.empty());
    
    // Should include CUDA
    EXPECT_NE(std::find(backends.begin(), backends.end(), BackendType::BT_CUDA),
              backends.end());
}

// =============================================================================
// Parameterized Strategy Tests
// =============================================================================

struct StrategyTestParam {
    BackendType requested;
    BackendType expected_effective;
    bool has_strategy;
};

class StrategyParameterizedTest : public ::testing::TestWithParam<StrategyTestParam> {};

TEST_P(StrategyParameterizedTest, FactoryBehavior) {
    auto param = GetParam();
    
    EXPECT_EQ(SearchStrategyFactory::get_effective_backend(param.requested),
              param.expected_effective);
    EXPECT_EQ(SearchStrategyFactory::has_strategy(param.requested),
              param.has_strategy);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    StrategyParameterizedTest,
    ::testing::Values(
        StrategyTestParam{BackendType::BT_CUDA, BackendType::BT_CUDA, true},
        StrategyTestParam{BackendType::BT_CUDNN, BackendType::BT_CUDA, true},
        StrategyTestParam{BackendType::BT_CUTLASS, BackendType::BT_CUDA, true},
        StrategyTestParam{BackendType::BT_TRITON, BackendType::BT_CUDA, true},
        StrategyTestParam{BackendType::BT_CPU, BackendType::BT_CPU, true},
        StrategyTestParam{BackendType::BT_MKL, BackendType::BT_CPU, true},
        StrategyTestParam{BackendType::BT_ROCM, BackendType::BT_ROCM, true},
        StrategyTestParam{BackendType::BT_MPS, BackendType::BT_MPS, true},
        StrategyTestParam{BackendType::BT_ASCEND, BackendType::BT_ASCEND, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
