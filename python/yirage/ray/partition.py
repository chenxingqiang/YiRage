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
Search space partitioning for distributed search.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import json


@dataclass
class SearchPartition:
    """
    Represents a partition of the search configuration space.

    Each partition can be explored independently by a worker.
    """

    partition_id: int = 0
    total_partitions: int = 1

    # Configuration ranges for this partition
    grid_dim_range: List[Tuple[int, int, int]] = field(default_factory=list)
    block_dim_range: List[Tuple[int, int, int]] = field(default_factory=list)
    imap_range: List[Tuple[int, int, int]] = field(default_factory=list)
    omap_range: List[Tuple[int, int, int]] = field(default_factory=list)
    fmap_range: List[int] = field(default_factory=list)
    frange_range: List[int] = field(default_factory=list)

    # Estimated work
    estimated_candidates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "partition_id": self.partition_id,
            "total_partitions": self.total_partitions,
            "grid_dim_range": [{"x": g[0], "y": g[1], "z": g[2]} for g in self.grid_dim_range],
            "block_dim_range": [{"x": b[0], "y": b[1], "z": b[2]} for b in self.block_dim_range],
            "imap_range": [{"x": i[0], "y": i[1], "z": i[2]} for i in self.imap_range],
            "omap_range": [{"x": o[0], "y": o[1], "z": o[2]} for o in self.omap_range],
            "fmap_range": self.fmap_range,
            "frange_range": self.frange_range,
            "estimated_candidates": self.estimated_candidates,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchPartition":
        """Create from dictionary."""
        return cls(
            partition_id=data.get("partition_id", 0),
            total_partitions=data.get("total_partitions", 1),
            grid_dim_range=[(g["x"], g["y"], g["z"]) for g in data.get("grid_dim_range", [])],
            block_dim_range=[(b["x"], b["y"], b["z"]) for b in data.get("block_dim_range", [])],
            imap_range=[(i["x"], i["y"], i["z"]) for i in data.get("imap_range", [])],
            omap_range=[(o["x"], o["y"], o["z"]) for o in data.get("omap_range", [])],
            fmap_range=data.get("fmap_range", []),
            frange_range=data.get("frange_range", []),
            estimated_candidates=data.get("estimated_candidates", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SearchPartition":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_partitions(
    num_partitions: int,
    grid_dims: Optional[List[Tuple[int, int, int]]] = None,
    block_dims: Optional[List[Tuple[int, int, int]]] = None,
    imaps: Optional[List[Tuple[int, int, int]]] = None,
    omaps: Optional[List[Tuple[int, int, int]]] = None,
    fmaps: Optional[List[int]] = None,
    franges: Optional[List[int]] = None,
) -> List[SearchPartition]:
    """
    Create search partitions from configuration space.

    Primary partitioning is by grid_dim since it typically has
    the most impact on search behavior.

    Args:
        num_partitions: Number of partitions to create
        grid_dims: Grid dimensions to explore
        block_dims: Block dimensions to explore
        imaps: Input mappings to explore
        omaps: Output mappings to explore
        fmaps: Forloop dimension mappings
        franges: Forloop ranges

    Returns:
        List of SearchPartition objects
    """
    # Defaults
    if grid_dims is None:
        grid_dims = [(1, 1, 1), (4, 1, 1), (8, 1, 1), (16, 1, 1)]
    if block_dims is None:
        block_dims = [(128, 1, 1), (256, 1, 1)]
    if imaps is None:
        imaps = []
    if omaps is None:
        omaps = []
    if fmaps is None:
        fmaps = [-1]
    if franges is None:
        franges = [1, 2, 4, 8]

    partitions = []

    # Partition by grid_dim when multiple grids; else block_dim (decode m=1)
    if len(grid_dims) > 1:
        dims_per_partition = max(
            1, (len(grid_dims) + num_partitions - 1) // num_partitions
        )
        for i in range(num_partitions):
            start_idx = i * dims_per_partition
            end_idx = min(start_idx + dims_per_partition, len(grid_dims))
            if start_idx >= len(grid_dims):
                partition = SearchPartition(
                    partition_id=i,
                    total_partitions=num_partitions,
                    estimated_candidates=0,
                )
            else:
                partition_grids = grid_dims[start_idx:end_idx]
                estimated = (
                    len(partition_grids)
                    * len(block_dims)
                    * max(1, len(imaps))
                    * max(1, len(omaps))
                    * max(1, len(franges))
                )
                partition = SearchPartition(
                    partition_id=i,
                    total_partitions=num_partitions,
                    grid_dim_range=partition_grids,
                    block_dim_range=block_dims,
                    imap_range=imaps,
                    omap_range=omaps,
                    fmap_range=fmaps,
                    frange_range=franges,
                    estimated_candidates=estimated,
                )
            partitions.append(partition)
        return partitions

    if len(block_dims) > 1:
        blocks_per_partition = max(
            1, (len(block_dims) + num_partitions - 1) // num_partitions
        )
        for i in range(num_partitions):
            start_idx = i * blocks_per_partition
            end_idx = min(start_idx + blocks_per_partition, len(block_dims))
            if start_idx >= len(block_dims):
                partition = SearchPartition(
                    partition_id=i,
                    total_partitions=num_partitions,
                    estimated_candidates=0,
                )
            else:
                partition_blocks = block_dims[start_idx:end_idx]
                estimated = (
                    len(grid_dims)
                    * len(partition_blocks)
                    * max(1, len(imaps))
                    * max(1, len(omaps))
                    * max(1, len(franges))
                )
                partition = SearchPartition(
                    partition_id=i,
                    total_partitions=num_partitions,
                    grid_dim_range=grid_dims,
                    block_dim_range=partition_blocks,
                    imap_range=imaps,
                    omap_range=omaps,
                    fmap_range=fmaps,
                    frange_range=franges,
                    estimated_candidates=estimated,
                )
            partitions.append(partition)
        return partitions

    # Single grid + single block — replicate full space to each worker
    for i in range(num_partitions):
        estimated = (
            len(grid_dims)
            * len(block_dims)
            * max(1, len(imaps))
            * max(1, len(omaps))
            * max(1, len(franges))
        )
        partitions.append(
            SearchPartition(
                partition_id=i,
                total_partitions=num_partitions,
                grid_dim_range=grid_dims,
                block_dim_range=block_dims,
                imap_range=imaps,
                omap_range=omaps,
                fmap_range=fmaps,
                frange_range=franges,
                estimated_candidates=estimated if i == 0 else 0,
            )
        )
    return partitions


def partition_config_from_search_config(
    search_config: Dict[str, Any],
    num_partitions: int,
) -> List[SearchPartition]:
    """
    Create partitions from a search configuration dictionary.

    Compatible with the config format used by superoptimize().
    """
    grid_dims = search_config.get("griddims") or search_config.get("grid_dims")
    block_dims = search_config.get("blockdims") or search_config.get("block_dims")
    imaps = search_config.get("imaps")
    omaps = search_config.get("omaps")
    fmaps = search_config.get("fmaps")
    franges = search_config.get("franges")

    return create_partitions(
        num_partitions=num_partitions,
        grid_dims=grid_dims,
        block_dims=block_dims,
        imaps=imaps,
        omaps=omaps,
        fmaps=fmaps,
        franges=franges,
    )
