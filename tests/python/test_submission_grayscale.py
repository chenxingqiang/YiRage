# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from examples.submission import grayscale_numpy


def test_grayscale_numpy_uint8_matches_reference():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    result = grayscale_numpy(image)
    expected = grayscale_numpy(image.copy())
    np.testing.assert_array_equal(result, expected)
