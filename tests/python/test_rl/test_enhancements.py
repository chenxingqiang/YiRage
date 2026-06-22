# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for all new RL enhancements:
- Problem 1: Bottom-up feedback (KernelCharacteristics)
- Problem 2: Layered reward shaping (LayeredRewardComputer)
- Problem 3: Batch search API (BatchSearchAPI)
- Problem 4: Dynamic features (DynamicFeatureDict)
- Problem 5: Surrogate model (SurrogateModel)
- Problem 6a: Z3 violation degree (Z3GuidedReward)
- Problem 6b: Expert demonstrations (ExpertGuidedReward)
- Problem 6c: Cross-backend migration (KernelMigrationEngine)
- Problem 6d: Persistent kernel (PersistentKernelSearchSpace)
"""

import numpy as np
import pytest


# =============================================================================
# Problem 1: KernelCharacteristics (Bottom-up Feedback)
# =============================================================================


class TestKernelCharacteristics:
    """Test bottom-up feedback from Level 2 to Level 0."""

    def test_default_creation(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        kc = KernelCharacteristics()
        assert kc.dominant_op_type == "unknown"
        assert kc.reuse_pattern == "none"
        assert kc.memory_intensity == 0.0
        assert kc.search_success_rate == 0.0

    def test_to_dict_from_dict(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        kc = KernelCharacteristics(
            dominant_op_type="matmul",
            reuse_pattern="weight_reuse",
            memory_intensity=15.0,
            compute_intensity=8.0,
            num_operators=12,
            num_matmuls=4,
            num_reductions=2,
            requires_large_shared_memory=True,
            requires_high_bandwidth=False,
            parallelism_degree=0.8,
            search_success_rate=0.65,
            common_failure_reason="buffer_overflow",
        )

        d = kc.to_dict()
        assert d["dominant_op_type"] == "matmul"
        assert d["memory_intensity"] == 15.0

        kc2 = KernelCharacteristics.from_dict(d)
        assert kc2.dominant_op_type == "matmul"
        assert kc2.reuse_pattern == "weight_reuse"
        assert kc2.memory_intensity == 15.0
        assert kc2.search_success_rate == 0.65

    def test_encode(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        kc = KernelCharacteristics(
            dominant_op_type="matmul",
            reuse_pattern="weight_reuse",
            memory_intensity=50.0,
            num_operators=10,
            search_success_rate=0.7,
        )
        encoded = kc.encode()
        assert encoded.shape == (12,)
        assert encoded.dtype == np.float32
        # Check non-zero values
        assert encoded[0] == pytest.approx(0.2)  # matmul
        assert encoded[1] == pytest.approx(0.25)  # weight_reuse
        assert encoded[10] == pytest.approx(0.7)  # search_success_rate

    def test_suggest_design_adjustments(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        # Memory-intensive kernel should suggest weight_stationary dataflow
        kc = KernelCharacteristics(
            memory_intensity=15.0,
            requires_large_shared_memory=True,
            search_success_rate=0.2,
            common_failure_reason="buffer_overflow",
        )
        suggestions = kc.suggest_design_adjustments()
        assert suggestions["increase_l1_buffer"] is True
        assert suggestions["min_l1_buffer_kb"] == 128.0
        assert suggestions["prefer_dataflow"] == "weight_stationary"
        assert suggestions["relax_constraints"] is True

    def test_suggest_no_adjustments_for_good_kernel(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        kc = KernelCharacteristics(
            memory_intensity=1.0,
            search_success_rate=0.9,
        )
        suggestions = kc.suggest_design_adjustments()
        assert len(suggestions) == 0

    def test_from_dict_ignores_unknown_fields(self):
        from yirage.rl.search.accelerator_space import KernelCharacteristics

        d = {
            "dominant_op_type": "matmul",
            "unknown_field": 42,
            "another_unknown": "value",
        }
        kc = KernelCharacteristics.from_dict(d)
        assert kc.dominant_op_type == "matmul"


class TestParetoPointWithKernelCharacteristics:
    """Test ParetoPoint with kernel characteristics field."""

    def test_pareto_point_with_kernel_characteristics(self):
        from yirage.rl.search.accelerator_space import ParetoPoint

        kc_dict = {"dominant_op_type": "matmul", "memory_intensity": 5.0}
        point = ParetoPoint(
            latency_ms=1.0,
            energy_pj=2.0,
            kernel_characteristics=kc_dict,
        )
        d = point.to_dict()
        assert "kernel_characteristics" in d
        assert d["kernel_characteristics"]["dominant_op_type"] == "matmul"

    def test_pareto_point_without_kernel_characteristics(self):
        from yirage.rl.search.accelerator_space import ParetoPoint

        point = ParetoPoint(latency_ms=1.0)
        d = point.to_dict()
        assert "kernel_characteristics" not in d

    def test_pareto_dominance_unchanged(self):
        from yirage.rl.search.accelerator_space import ParetoPoint

        p1 = ParetoPoint(latency_ms=1.0, energy_pj=1.0, area_mm2=1.0, power_mw=1.0)
        p2 = ParetoPoint(latency_ms=2.0, energy_pj=2.0, area_mm2=2.0, power_mw=2.0)
        assert p1.dominates(p2)
        assert not p2.dominates(p1)


class TestAcceleratorEnvKernelFeedback:
    """Test AcceleratorEnv with bottom-up kernel feedback."""

    def test_set_level1_result_with_kernel_characteristics(self):
        from yirage.rl.search.accelerator_space import AcceleratorEnv

        env = AcceleratorEnv()
        result = {
            "verified": True,
            "latency_ms": 2.0,
            "kernel_characteristics": {
                "dominant_op_type": "matmul",
                "memory_intensity": 10.0,
                "search_success_rate": 0.7,
            },
        }
        env.set_level1_result(result)

        assert env.kernel_feedback is not None
        assert env.kernel_feedback.dominant_op_type == "matmul"
        assert env.kernel_feedback.memory_intensity == 10.0

    def test_observation_includes_kernel_feedback(self):
        from yirage.rl.search.accelerator_space import AcceleratorEnv

        env = AcceleratorEnv()
        obs, _ = env.reset()

        # Should have kernel_feedback key
        assert "kernel_feedback" in obs
        assert obs["kernel_feedback"].shape == (12,)

        # Initially zeros (no feedback yet)
        assert np.all(obs["kernel_feedback"] == 0.0)

    def test_reset_clears_kernel_feedback(self):
        from yirage.rl.search.accelerator_space import (
            AcceleratorEnv,
            KernelCharacteristics,
        )

        env = AcceleratorEnv()
        env.kernel_feedback = KernelCharacteristics(dominant_op_type="matmul")
        env.reset()
        assert env.kernel_feedback is None


# =============================================================================
# Problem 2: Layered Reward Shaping
# =============================================================================


class TestLayeredRewardComputer:
    """Test per-level reward decomposition."""

    def test_level0_reward_utilization_based(self):
        from yirage.rl.env.reward import LayeredRewardComputer

        lrc = LayeredRewardComputer()

        # High utilization → high reward
        reward_good = lrc.compute_level0_reward(
            pe_utilization=0.9,
            buffer_utilization=0.8,
            noc_utilization=0.7,
            level2_success_rate=0.6,
        )

        # Low utilization → low reward
        reward_bad = lrc.compute_level0_reward(
            pe_utilization=0.1,
            buffer_utilization=0.1,
            noc_utilization=0.1,
            level2_success_rate=0.0,
        )

        assert reward_good > reward_bad
        assert reward_good > 0

    def test_level1_reward_feasibility_based(self):
        from yirage.rl.env.reward import LayeredRewardComputer

        lrc = LayeredRewardComputer()

        # All configs valid
        reward_good = lrc.compute_level1_reward(
            num_valid_configs=10,
            num_total_configs=10,
            config_diversity=0.8,
            constraint_satisfaction=0.9,
        )

        # No configs valid
        reward_bad = lrc.compute_level1_reward(
            num_valid_configs=0,
            num_total_configs=10,
            config_diversity=0.0,
            constraint_satisfaction=0.0,
        )

        assert reward_good > reward_bad

    def test_level2_reward_performance_based(self):
        from yirage.rl.env.reward import (
            LayeredRewardComputer,
            VerifyResult,
            ProfileResult,
        )

        lrc = LayeredRewardComputer()

        # Good kernel
        reward_good = lrc.compute_level2_reward(
            verify_result=VerifyResult(verified=True, fingerprint_time_ms=0.1),
            profile_result=ProfileResult(latency_ms=0.5),
            baseline_latency_ms=10.0,
        )

        # Invalid kernel
        reward_bad = lrc.compute_level2_reward(
            verify_result=VerifyResult(verified=False, fingerprint_time_ms=0.1),
            profile_result=None,
        )

        assert reward_good > reward_bad

    def test_level2_reward_with_z3_violation(self):
        from yirage.rl.env.reward import (
            LayeredRewardComputer,
            Z3ViolationInfo,
        )

        lrc = LayeredRewardComputer()

        # Mostly satisfied
        reward_good = lrc.compute_level2_reward(
            z3_violation=Z3ViolationInfo(
                total_constraints=10,
                satisfied_constraints=9,
                violated_constraints=1,
                mean_violation_degree=0.1,
            ),
        )

        # Mostly violated
        reward_bad = lrc.compute_level2_reward(
            z3_violation=Z3ViolationInfo(
                total_constraints=10,
                satisfied_constraints=2,
                violated_constraints=8,
                mean_violation_degree=0.8,
            ),
        )

        assert reward_good > reward_bad

    def test_get_stats(self):
        from yirage.rl.env.reward import LayeredRewardComputer

        lrc = LayeredRewardComputer()
        lrc.compute_level0_reward(0.5, 0.5, 0.5, 0.5)
        lrc.compute_level1_reward(5, 10, 0.5, 0.5)
        lrc.compute_level2_reward()

        stats = lrc.get_stats()
        assert stats["level0"]["total"] == 1
        assert stats["level1"]["total"] == 1
        assert stats["level2"]["total"] == 1

    def test_reset(self):
        from yirage.rl.env.reward import LayeredRewardComputer

        lrc = LayeredRewardComputer()
        lrc.compute_level0_reward(0.5, 0.5, 0.5, 0.5)
        lrc.reset()
        assert lrc.level0_stats["total"] == 0


# =============================================================================
# Problem 6a: Z3 Violation Degree
# =============================================================================


class TestZ3ViolationInfo:
    """Test Z3 violation degree as continuous reward."""

    def test_satisfaction_ratio(self):
        from yirage.rl.env.reward import Z3ViolationInfo

        info = Z3ViolationInfo(
            total_constraints=10,
            satisfied_constraints=7,
            violated_constraints=3,
        )
        assert info.satisfaction_ratio == pytest.approx(0.7)

    def test_satisfaction_ratio_empty(self):
        from yirage.rl.env.reward import Z3ViolationInfo

        info = Z3ViolationInfo(total_constraints=0)
        assert info.satisfaction_ratio == 0.0

    def test_to_dict_from_dict(self):
        from yirage.rl.env.reward import Z3ViolationInfo

        info = Z3ViolationInfo(
            total_constraints=10,
            satisfied_constraints=8,
            violated_constraints=2,
            mean_violation_degree=0.15,
            max_violation_degree=0.3,
        )
        d = info.to_dict()
        assert d["satisfaction_ratio"] == pytest.approx(0.8)

        info2 = Z3ViolationInfo.from_dict(d)
        assert info2.total_constraints == 10
        assert info2.mean_violation_degree == pytest.approx(0.15)


class TestZ3GuidedReward:
    """Test Z3-guided continuous reward."""

    def test_reward_higher_with_high_satisfaction(self):
        from yirage.rl.env.reward import Z3GuidedReward, Z3ViolationInfo

        rgr = Z3GuidedReward()

        reward_good = rgr.compute(
            verify_result=None,
            profile_result=None,
            config_hash="a",
            search_depth=1,
            action_type=0,
            z3_violation=Z3ViolationInfo(
                total_constraints=10,
                satisfied_constraints=9,
                violated_constraints=1,
                mean_violation_degree=0.05,
            ),
        )

        reward_bad = rgr.compute(
            verify_result=None,
            profile_result=None,
            config_hash="b",
            search_depth=1,
            action_type=0,
            z3_violation=Z3ViolationInfo(
                total_constraints=10,
                satisfied_constraints=2,
                violated_constraints=8,
                mean_violation_degree=0.8,
            ),
        )

        assert reward_good > reward_bad

    def test_reward_without_z3_matches_base(self):
        from yirage.rl.env.reward import Z3GuidedReward, RewardComputer

        z3r = Z3GuidedReward()
        base = RewardComputer()

        # Without Z3 info, should behave like base
        r_z3 = z3r.compute(None, None, "c", 1, 0)
        r_base = base.compute(None, None, "c", 1, 0)

        # Should be close (both use exploration bonus for new config)
        assert abs(r_z3 - r_base) < 0.01


# =============================================================================
# Problem 6b: Expert Demonstrations
# =============================================================================


class TestExpertGuidedReward:
    """Test expert demonstration reward."""

    def test_imitation_bonus_when_matching_expert(self):
        from yirage.rl.env.reward import ExpertGuidedReward

        egr = ExpertGuidedReward()
        egr.add_expert_trajectory({
            "actions": [
                {"action_type": 0, "operator": "matmul"},
                {"action_type": 2, "operator": "add"},
            ],
            "latency_ms": 1.0,
        })

        # Action matching expert at depth 0
        reward_match = egr.compute(
            verify_result=None,
            profile_result=None,
            config_hash="d",
            search_depth=0,
            action_type=0,
            current_action={"action_type": 0, "operator": "matmul"},
        )

        # Action not matching expert
        reward_no_match = egr.compute(
            verify_result=None,
            profile_result=None,
            config_hash="e",
            search_depth=0,
            action_type=0,
            current_action={"action_type": 0, "operator": "softmax"},
        )

        assert reward_match > reward_no_match

    def test_imitation_decay(self):
        from yirage.rl.env.reward import ExpertGuidedReward

        egr = ExpertGuidedReward()
        initial = egr.current_imitation_bonus

        for _ in range(100):
            egr.decay_imitation()

        assert egr.current_imitation_bonus < initial
        assert egr.current_imitation_bonus >= egr.expert_config.min_imitation_bonus


# =============================================================================
# Problem 3: Batch Search API
# =============================================================================


class TestBatchSearchAPI:
    """Test batch search interface."""

    def test_batch_search_empty(self):
        from yirage.rl.search.batch_search import BatchSearchAPI

        api = BatchSearchAPI()
        results = api.search_batch("{}", [])
        assert results == []

    def test_batch_search_multiple_configs(self):
        from yirage.rl.search.batch_search import BatchSearchAPI

        api = BatchSearchAPI()
        configs = [
            {"grid_dim_x": 1, "block_dim_x": 128},
            {"grid_dim_x": 4, "block_dim_x": 256},
            {"grid_dim_x": 8, "block_dim_x": 512},
        ]
        results = api.search_batch("{}", configs)
        assert len(results) == 3
        assert all(r.config_id == i for i, r in enumerate(results))

    def test_kernel_search_result_to_dict(self):
        from yirage.rl.search.batch_search import KernelSearchResult

        r = KernelSearchResult(
            config_id=0,
            verified=True,
            latency_ms=1.5,
            search_time_ms=10.0,
        )
        d = r.to_dict()
        assert d["verified"] is True
        assert d["latency_ms"] == 1.5

    def test_perturb_config(self):
        from yirage.rl.search.batch_search import BatchSearchAPI

        config = {"grid_dim_x": 8, "block_dim_x": 256, "forloop_range": 4}
        perturbed = BatchSearchAPI._perturb_config(config, n_perturbations=5)
        assert len(perturbed) == 5
        # All should be different dicts
        for p in perturbed:
            assert "grid_dim_x" in p
            assert p["grid_dim_x"] >= 1

    def test_search_with_expert_warmstart(self):
        from yirage.rl.search.batch_search import BatchSearchAPI

        api = BatchSearchAPI()
        configs = [{"grid_dim_x": 1, "block_dim_x": 128}]
        expert_results = [
            {"config": {"grid_dim_x": 4, "block_dim_x": 256, "forloop_range": 8}},
        ]
        results = api.search_with_expert_warmstart("{}", configs, expert_results)
        # Should have original + expert perturbations
        assert len(results) > len(configs)


# =============================================================================
# Problem 4: Dynamic Features
# =============================================================================


class TestDynamicFeatureDict:
    """Test extensible feature dictionary."""

    def test_set_and_get(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        dfd.set("test_feature", np.array([1.0, 2.0, 3.0]))

        result = dfd.get("test_feature")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_get_missing_uses_registry(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        # "graph_topology" is in FEATURE_REGISTRY with dim=8
        result = dfd.get("graph_topology")
        assert result.shape == (8,)
        assert np.all(result == 0.0)

    def test_get_missing_with_default_dim(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        result = dfd.get("unknown_feature", default_dim=5)
        assert result.shape == (5,)

    def test_to_flat_vector(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        dfd.set("a", np.array([1.0, 2.0]))
        dfd.set("b", np.array([3.0, 4.0, 5.0]))

        flat = dfd.to_flat_vector(feature_order=["a", "b"])
        np.testing.assert_array_equal(flat, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_to_flat_vector_with_missing(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        dfd.set("a", np.array([1.0, 2.0]))

        flat = dfd.to_flat_vector(feature_order=["a", "graph_topology"])
        assert len(flat) == 2 + 8  # a(2) + graph_topology(8)

    def test_to_dict_from_dict(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        dfd.set("x", np.array([1.0, 2.0]))
        dfd.set("y", np.array([3.0]))

        d = dfd.to_dict()
        assert d["x"] == [1.0, 2.0]

        dfd2 = DynamicFeatureDict.from_dict(d)
        np.testing.assert_array_equal(dfd2.get("x"), [1.0, 2.0])

    def test_keys(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureDict

        dfd = DynamicFeatureDict()
        dfd.set("a", np.array([1.0]))
        dfd.set("b", np.array([2.0]))
        assert dfd.keys() == {"a", "b"}


class TestDynamicFeatureProcessor:
    """Test dynamic feature processing from MuGraphFeature."""

    def test_process_empty_graph(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureProcessor
        from yirage.rl.features.mugraph_features import MuGraphFeature

        proc = DynamicFeatureProcessor()
        features = MuGraphFeature()
        result = proc.process(features)

        assert "graph_topology" in result.keys()
        assert "operator_histogram" in result.keys()
        assert "hardware_config" in result.keys()
        assert "search_state" in result.keys()

    def test_process_with_operators(self):
        from yirage.rl.features.dynamic_features import DynamicFeatureProcessor
        from yirage.rl.features.mugraph_features import (
            MuGraphFeature,
            OperatorFeature,
            TensorFeature,
        )

        features = MuGraphFeature(
            operators=[
                OperatorFeature(op_id=0, op_type="matmul", num_inputs=2, num_outputs=1),
                OperatorFeature(op_id=1, op_type="add", num_inputs=2, num_outputs=1),
            ],
            tensors=[TensorFeature(tensor_id=0, dims=[128, 128])],
            num_operators=2,
            num_tensors=1,
            grid_dim=(4, 1, 1),
            block_dim=(128, 1, 1),
        )

        proc = DynamicFeatureProcessor()
        result = proc.process(features)

        # Operator histogram should have non-zero values
        hist = result.get("operator_histogram")
        assert hist[0] > 0  # matmul
        assert hist[1] > 0  # add


# =============================================================================
# Problem 5: Surrogate Model
# =============================================================================


class TestSurrogateModel:
    """Test learned surrogate model for AccelForge."""

    def test_predict_correction_default(self):
        from yirage.rl.hardware.surrogate_model import SurrogateModel

        model = SurrogateModel()
        lat, eng, area = model.predict_correction(
            {"pe_array_rows": 16, "pe_array_cols": 16},
        )
        # Should be in [0.5, 2.0] range
        assert 0.5 <= lat <= 2.0
        assert 0.5 <= eng <= 2.0
        assert 0.5 <= area <= 2.0

    def test_add_calibration_and_update(self):
        from yirage.rl.hardware.surrogate_model import (
            SurrogateModel,
            CalibrationPoint,
        )

        model = SurrogateModel()

        # Add calibration data
        for i in range(15):
            model.add_calibration(CalibrationPoint(
                design={"pe_array_rows": 16, "pe_array_cols": 16},
                predicted_latency_ms=2.0,
                actual_latency_ms=2.5,
                predicted_energy_pj=5.0,
                actual_energy_pj=6.0,
                predicted_area_mm2=10.0,
                actual_area_mm2=10.0,
            ))

        assert model.n_calibrations == 15
        assert model.mean_latency_error > 0

    def test_encode_design(self):
        from yirage.rl.hardware.surrogate_model import SurrogateModel

        model = SurrogateModel()
        features = model.encode_design({
            "pe_array_rows": 32,
            "pe_array_cols": 32,
            "l0_buffer_kb": 2.0,
            "l1_buffer_kb": 64.0,
            "l2_buffer_kb": 512.0,
            "dataflow": "row_stationary",
            "noc_topology": "mesh",
            "data_precision": "fp16",
            "clock_mhz": 1000.0,
            "tech_node_nm": 7,
        })
        assert features.shape == (10,)
        assert np.all(np.isfinite(features))

    def test_encode_workload(self):
        from yirage.rl.hardware.surrogate_model import SurrogateModel

        model = SurrogateModel()
        features = model.encode_workload({
            "estimated_flops": 1e9,
            "batch_size": 32,
            "sequence_length": 2048,
        })
        assert features.shape == (10,)

    def test_get_stats(self):
        from yirage.rl.hardware.surrogate_model import SurrogateModel

        model = SurrogateModel()
        stats = model.get_stats()
        assert stats["n_calibrations"] == 0
        assert stats["calibration_data_size"] == 0

    def test_save_load(self, tmp_path):
        from yirage.rl.hardware.surrogate_model import (
            SurrogateModel,
            CalibrationPoint,
        )

        model = SurrogateModel()
        model.add_calibration(CalibrationPoint(
            design={"pe_array_rows": 8},
            predicted_latency_ms=1.0,
            actual_latency_ms=1.2,
        ))

        save_path = str(tmp_path / "surrogate")
        model.save(save_path)

        loaded = SurrogateModel.load(save_path)
        assert loaded.n_calibrations == 1
        np.testing.assert_array_equal(loaded.w1, model.w1)


# =============================================================================
# Problem 6c: Cross-Backend Migration
# =============================================================================


class TestKernelMigrationEngine:
    """Test cross-backend kernel migration."""

    def test_cuda_to_rocm_feasible(self):
        from yirage.rl.search.cross_backend import KernelMigrationEngine

        engine = KernelMigrationEngine()
        # Use block_dim_x=100, not divisible by rocm warp_size=64
        config = {
            "block_dim_x": 100,
            "block_dim_y": 1,
            "shared_memory_size": 32768,  # 32KB
        }
        result = engine.check_migration_feasibility(config, "cuda", "rocm")
        assert result.feasible is True
        # Different warp size should require adaptation (100 % 64 != 0)
        assert any("adjust_block_x" in a for a in result.adaptations)

    def test_cuda_to_cpu_infeasible_shared_mem(self):
        from yirage.rl.search.cross_backend import KernelMigrationEngine

        engine = KernelMigrationEngine()
        config = {
            "block_dim_x": 128,
            "block_dim_y": 1,
            "shared_memory_size": 32768,  # 32KB, CPU has no shared mem
        }
        result = engine.check_migration_feasibility(config, "cuda", "cpu")
        assert result.feasible is False
        assert "target_no_shared_memory" in result.blockers

    def test_cuda_to_ascend_thread_limit(self):
        from yirage.rl.search.cross_backend import KernelMigrationEngine

        engine = KernelMigrationEngine()
        config = {
            "block_dim_x": 512,  # Ascend max is 256
            "block_dim_y": 1,
            "shared_memory_size": 0,
        }
        result = engine.check_migration_feasibility(config, "cuda", "ascend")
        assert result.feasible is False

    def test_get_migration_targets(self):
        from yirage.rl.search.cross_backend import KernelMigrationEngine

        engine = KernelMigrationEngine()
        config = {"block_dim_x": 64, "block_dim_y": 1, "shared_memory_size": 0}
        targets = engine.get_migration_targets(config, "cuda")
        assert "rocm" in targets
        assert "maca" in targets
        assert "cuda" not in targets  # Source excluded

    def test_adapt_kernel(self):
        from yirage.rl.search.cross_backend import (
            KernelMigrationEngine,
            MigrationResult,
        )

        engine = KernelMigrationEngine()
        config = {"block_dim_x": 128, "uses_tensor_cores": True}
        migration = MigrationResult(
            source_backend="cuda",
            target_backend="rocm",
            feasible=True,
            adaptations=["adjust_block_x: 128 → 192", "replace_tensor_core_ops"],
        )
        adapted = engine.adapt_kernel(config, migration)
        assert adapted["uses_tensor_cores"] is False
        assert adapted["target_backend"] == "rocm"

    def test_migration_result_to_dict(self):
        from yirage.rl.search.cross_backend import MigrationResult

        r = MigrationResult(
            source_backend="cuda",
            target_backend="rocm",
            feasible=True,
            adaptations=["adjust_block_x"],
        )
        d = r.to_dict()
        assert d["source_backend"] == "cuda"
        assert d["feasible"] is True


# =============================================================================
# Problem 6d: Persistent Kernel Integration
# =============================================================================


class TestPersistentKernelSearchSpace:
    """Test persistent kernel RL integration."""

    def test_get_persistent_action_mask(self):
        from yirage.rl.search.cross_backend import PersistentKernelSearchSpace

        pkss = PersistentKernelSearchSpace()

        # Incomplete kernel — can't make persistent
        mask = pkss.get_persistent_action_mask(0, 0, "cuda")
        assert mask[0] == 0

        # Complete kernel on CUDA — can make persistent
        mask = pkss.get_persistent_action_mask(3, 2, "cuda")
        assert mask[0] == 1  # cooperative groups supported
        assert mask[1] == 1
        assert mask[2] == 1

    def test_compute_persistent_reward(self):
        from yirage.rl.search.cross_backend import (
            PersistentKernelSearchSpace,
            PersistentKernelConfig,
        )

        pkss = PersistentKernelSearchSpace()

        config = PersistentKernelConfig(persistent=True, queue_depth=4)
        reward = pkss.compute_persistent_reward(
            config,
            standard_latency_ms=1.0,
            launch_overhead_us=5.0,
            num_inferences=1000,
        )
        assert reward > 0  # Persistent should save time

        config_no = PersistentKernelConfig(persistent=False)
        reward_no = pkss.compute_persistent_reward(config_no, 1.0)
        assert reward_no == 0.0

    def test_suggest_persistent_config(self):
        from yirage.rl.search.cross_backend import PersistentKernelSearchSpace

        pkss = PersistentKernelSearchSpace()
        config = pkss.suggest_persistent_config(
            {"grid_dim_x": 8, "grid_dim_y": 4},
            workload_type="decode",
        )
        assert config.persistent is True
        assert config.num_persistent_blocks == 32
        assert config.workload_type == "decode"
        assert config.max_duration_us == 10000

    def test_persistent_kernel_config_to_dict_from_dict(self):
        from yirage.rl.search.cross_backend import PersistentKernelConfig

        config = PersistentKernelConfig(
            persistent=True,
            max_duration_us=5000,
            queue_depth=8,
        )
        d = config.to_dict()
        assert d["persistent"] is True

        config2 = PersistentKernelConfig.from_dict(d)
        assert config2.persistent is True
        assert config2.max_duration_us == 5000


# =============================================================================
# Integration: Module imports
# =============================================================================


class TestModuleImports:
    """Test that all new classes are properly exported."""

    def test_import_kernel_characteristics(self):
        from yirage.rl import KernelCharacteristics
        assert KernelCharacteristics is not None

    def test_import_batch_search(self):
        from yirage.rl import BatchSearchAPI, BatchSearchConfig
        assert BatchSearchAPI is not None
        assert BatchSearchConfig is not None

    def test_import_dynamic_features(self):
        from yirage.rl import DynamicFeatureDict, DynamicFeatureProcessor
        assert DynamicFeatureDict is not None
        assert DynamicFeatureProcessor is not None

    def test_import_surrogate_model(self):
        from yirage.rl import SurrogateModel, CalibrationPoint
        assert SurrogateModel is not None
        assert CalibrationPoint is not None

    def test_import_migration_engine(self):
        from yirage.rl import KernelMigrationEngine, MigrationResult
        assert KernelMigrationEngine is not None
        assert MigrationResult is not None

    def test_import_persistent_kernel(self):
        from yirage.rl import PersistentKernelConfig, PersistentKernelSearchSpace
        assert PersistentKernelConfig is not None
        assert PersistentKernelSearchSpace is not None

    def test_import_z3_reward(self):
        from yirage.rl.env.reward import (
            Z3GuidedReward,
            Z3ViolationInfo,
            ExpertGuidedReward,
            LayeredRewardComputer,
        )
        assert Z3GuidedReward is not None
        assert Z3ViolationInfo is not None
        assert ExpertGuidedReward is not None
        assert LayeredRewardComputer is not None
