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
Feature extraction and processing for RL models.

Features flow from C++ µGraph to Python RL models:

    C++ µGraph → GraphFeatureExtractor → FeatureProcessor → PolicyNetwork

Enhanced features:
- DynamicFeatureDict: Extensible feature container (Problem 4)
- DynamicFeatureProcessor: Key-based feature processing
"""

from .mugraph_features import (
    OperatorFeature,
    TensorFeature,
    MuGraphFeature,
)

from .processor import (
    FeatureProcessor,
    FeatureNormalizer,
)

from .extractor import (
    GraphFeatureExtractor,
)

from .dynamic_features import (
    DynamicFeatureDict,
    DynamicFeatureProcessor,
    FeatureSpec,
    FEATURE_REGISTRY,
)

__all__ = [
    # Feature dataclasses
    "OperatorFeature",
    "TensorFeature",
    "MuGraphFeature",
    # Processing
    "FeatureProcessor",
    "FeatureNormalizer",
    # Extraction
    "GraphFeatureExtractor",
    # Dynamic Features (Problem 4)
    "DynamicFeatureDict",
    "DynamicFeatureProcessor",
    "FeatureSpec",
    "FEATURE_REGISTRY",
]
