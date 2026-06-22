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
Search feedback data structures for RL training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json


@dataclass
class CandidateInfo:
    """
    Information about a candidate configuration explored during search.

    This data is collected for RL training.
    """

    candidate_id: int = 0

    # Configuration
    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (128, 1, 1)
    imaps: List[Tuple[int, int, int]] = field(default_factory=list)
    omap: Tuple[int, int, int] = (0, 0, 0)
    frange: int = 1

    # Search context
    search_depth: int = 0
    operator_count: int = 0

    # Evaluation results
    verified: bool = False
    fingerprint_time_ms: float = 0.0
    estimated_performance_ms: float = 0.0
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "grid_dim": list(self.grid_dim),
            "block_dim": list(self.block_dim),
            "imaps": [list(m) for m in self.imaps],
            "omap": list(self.omap),
            "frange": self.frange,
            "search_depth": self.search_depth,
            "operator_count": self.operator_count,
            "verified": self.verified,
            "fingerprint_time_ms": self.fingerprint_time_ms,
            "estimated_performance_ms": self.estimated_performance_ms,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateInfo":
        """Create from dictionary."""
        return cls(
            candidate_id=data.get("candidate_id", 0),
            grid_dim=tuple(data.get("grid_dim", [1, 1, 1])),
            block_dim=tuple(data.get("block_dim", [128, 1, 1])),
            imaps=[tuple(m) for m in data.get("imaps", [])],
            omap=tuple(data.get("omap", [0, 0, 0])),
            frange=data.get("frange", 1),
            search_depth=data.get("search_depth", 0),
            operator_count=data.get("operator_count", 0),
            verified=data.get("verified", False),
            fingerprint_time_ms=data.get("fingerprint_time_ms", 0.0),
            estimated_performance_ms=data.get("estimated_performance_ms", 0.0),
            rejection_reason=data.get("rejection_reason", ""),
        )


@dataclass
class SearchFeedback:
    """
    Aggregated feedback from a search run.

    Contains all information needed for:
    1. Analyzing search efficiency
    2. Training RL search policies
    3. Debugging search issues
    """

    partition_id: int = 0
    total_partitions: int = 1

    candidates: List[CandidateInfo] = field(default_factory=list)
    valid_candidate_ids: List[int] = field(default_factory=list)

    # Statistics
    total_states_explored: int = 0
    valid_graphs_found: int = 0
    candidates_generated: int = 0

    # Timing
    search_time_seconds: float = 0.0

    # Best result
    best_performance_ms: float = float("inf")
    best_candidate_id: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_id": self.partition_id,
            "total_partitions": self.total_partitions,
            "candidates": [c.to_dict() for c in self.candidates],
            "valid_candidate_ids": self.valid_candidate_ids,
            "total_states_explored": self.total_states_explored,
            "valid_graphs_found": self.valid_graphs_found,
            "candidates_generated": self.candidates_generated,
            "search_time_seconds": self.search_time_seconds,
            "best_performance_ms": self.best_performance_ms,
            "best_candidate_id": self.best_candidate_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchFeedback":
        """Create from dictionary."""
        return cls(
            partition_id=data.get("partition_id", 0),
            total_partitions=data.get("total_partitions", 1),
            candidates=[CandidateInfo.from_dict(c) for c in data.get("candidates", [])],
            valid_candidate_ids=data.get("valid_candidate_ids", []),
            total_states_explored=data.get("total_states_explored", 0),
            valid_graphs_found=data.get("valid_graphs_found", 0),
            candidates_generated=data.get("candidates_generated", 0),
            search_time_seconds=data.get("search_time_seconds", 0.0),
            best_performance_ms=data.get("best_performance_ms", float("inf")),
            best_candidate_id=data.get("best_candidate_id", -1),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SearchFeedback":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def merge(cls, feedbacks: List["SearchFeedback"]) -> "SearchFeedback":
        """Merge feedback from multiple partitions."""
        if not feedbacks:
            return cls()

        merged = cls(
            partition_id=-1,  # Merged
            total_partitions=feedbacks[0].total_partitions,
        )

        candidate_offset = 0

        for fb in feedbacks:
            # Merge candidates with offset IDs
            for c in fb.candidates:
                new_c = CandidateInfo.from_dict(c.to_dict())
                new_c.candidate_id += candidate_offset
                merged.candidates.append(new_c)

            # Merge valid IDs with offset
            for vid in fb.valid_candidate_ids:
                merged.valid_candidate_ids.append(vid + candidate_offset)

            candidate_offset += len(fb.candidates)

            # Aggregate statistics
            merged.total_states_explored += fb.total_states_explored
            merged.valid_graphs_found += fb.valid_graphs_found
            merged.candidates_generated += fb.candidates_generated

            # Max time (parallel)
            merged.search_time_seconds = max(merged.search_time_seconds, fb.search_time_seconds)

            # Best result
            if fb.best_performance_ms < merged.best_performance_ms:
                merged.best_performance_ms = fb.best_performance_ms
                merged.best_candidate_id = (
                    fb.best_candidate_id + candidate_offset - len(fb.candidates)
                )

        return merged

    def get_summary(self) -> str:
        """Get summary string."""
        return (
            f"=== Search Feedback Summary ===\n"
            f"Partition: {self.partition_id}/{self.total_partitions}\n"
            f"States explored: {self.total_states_explored}\n"
            f"Candidates generated: {self.candidates_generated}\n"
            f"Valid graphs found: {self.valid_graphs_found}\n"
            f"Search time: {self.search_time_seconds:.2f}s\n"
            f"Best performance: {self.best_performance_ms:.3f}ms\n"
        )


@dataclass
class TrainingSample:
    """
    Training sample for RL (state, action, reward, next_state, done).
    """

    # State
    state_search_depth: int = 0
    state_operator_count: int = 0
    state_grid_dim: List[int] = field(default_factory=lambda: [1, 1, 1])
    state_block_dim: List[int] = field(default_factory=lambda: [128, 1, 1])
    state_num_valid_found: int = 0

    # Action
    action_imaps: List[List[int]] = field(default_factory=list)
    action_omap: List[int] = field(default_factory=lambda: [0, 0, 0])
    action_frange: int = 1

    # Reward
    reward: float = 0.0

    # Next state (None if terminal)
    next_state: Optional[Dict] = None

    # Terminal
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": {
                "search_depth": self.state_search_depth,
                "operator_count": self.state_operator_count,
                "grid_dim": self.state_grid_dim,
                "block_dim": self.state_block_dim,
                "num_valid_found": self.state_num_valid_found,
            },
            "action": {
                "imaps": self.action_imaps,
                "omap": self.action_omap,
                "frange": self.action_frange,
            },
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
        }


def extract_training_samples(
    feedback: SearchFeedback,
    validity_reward: float = 1.0,
    invalid_penalty: float = -0.5,
    depth_penalty: float = -0.01,
) -> List[TrainingSample]:
    """
    Extract training samples from search feedback.

    Args:
        feedback: Search feedback data
        validity_reward: Reward for verified candidates
        invalid_penalty: Penalty for invalid candidates
        depth_penalty: Penalty per search depth level

    Returns:
        List of training samples
    """
    samples = []
    num_valid_found = 0

    for i, cand in enumerate(feedback.candidates):
        sample = TrainingSample()

        # State
        sample.state_search_depth = cand.search_depth
        sample.state_operator_count = cand.operator_count
        sample.state_grid_dim = list(cand.grid_dim)
        sample.state_block_dim = list(cand.block_dim)
        sample.state_num_valid_found = num_valid_found

        # Action
        sample.action_imaps = [list(m) for m in cand.imaps]
        sample.action_omap = list(cand.omap)
        sample.action_frange = cand.frange

        # Reward
        sample.reward = depth_penalty * cand.search_depth
        if cand.verified:
            sample.reward += validity_reward
            if cand.estimated_performance_ms > 0:
                sample.reward += 1.0 / cand.estimated_performance_ms
            num_valid_found += 1
        else:
            sample.reward += invalid_penalty

        # Next state
        if i + 1 < len(feedback.candidates):
            next_cand = feedback.candidates[i + 1]
            sample.next_state = {
                "search_depth": next_cand.search_depth,
                "operator_count": next_cand.operator_count,
                "grid_dim": list(next_cand.grid_dim),
                "block_dim": list(next_cand.block_dim),
            }
            sample.done = False
        else:
            sample.next_state = None
            sample.done = True

        samples.append(sample)

    return samples


def save_training_data(
    feedbacks: List[SearchFeedback],
    filepath: str,
    **kwargs,
):
    """
    Save training data extracted from search feedbacks.

    Args:
        feedbacks: List of search feedbacks
        filepath: Output file path
        **kwargs: Parameters for extract_training_samples
    """
    all_samples = []

    for fb in feedbacks:
        samples = extract_training_samples(fb, **kwargs)
        all_samples.extend([s.to_dict() for s in samples])

    data = {
        "num_samples": len(all_samples),
        "samples": all_samples,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(all_samples)} training samples to {filepath}")
