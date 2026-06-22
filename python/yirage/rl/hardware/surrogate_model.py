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
Surrogate Model for AccelForge (Problem 5)

Learns a neural network approximation of hardware performance from
calibration data (real GPU profiling or cycle-accurate simulator).

This replaces the analytical model with a learned model that:
- Captures NoC congestion effects
- Models buffer spilling costs
- Accounts for compiler optimization impact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path


@dataclass
class CalibrationPoint:
    """A single calibration data point from real hardware or simulator."""

    design: Dict[str, Any] = field(default_factory=dict)
    workload: Dict[str, Any] = field(default_factory=dict)
    # Predicted by analytical model
    predicted_latency_ms: float = 0.0
    predicted_energy_pj: float = 0.0
    predicted_area_mm2: float = 0.0
    # Measured from real hardware/simulator
    actual_latency_ms: float = 0.0
    actual_energy_pj: float = 0.0
    actual_area_mm2: float = 0.0
    # Error
    latency_error_pct: float = 0.0
    energy_error_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design": self.design,
            "workload": self.workload,
            "predicted_latency_ms": self.predicted_latency_ms,
            "predicted_energy_pj": self.predicted_energy_pj,
            "predicted_area_mm2": self.predicted_area_mm2,
            "actual_latency_ms": self.actual_latency_ms,
            "actual_energy_pj": self.actual_energy_pj,
            "actual_area_mm2": self.actual_area_mm2,
        }


class SurrogateModel:
    """
    Learned surrogate model for AccelForge hardware performance prediction.

    Maintains a dataset of (design, workload) → (latency, energy, area)
    calibration points and learns a correction model on top of the
    analytical model.

    Architecture:
        analytical_prediction + NN_correction(design_features, workload_features)

    This allows:
    - Fast inference (~μs, same as analytical)
    - Better accuracy (captures non-linear effects)
    - Online learning (update with new calibration data)
    """

    def __init__(self, hidden_dim: int = 64, learning_rate: float = 1e-3, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.seed = seed

        # Calibration dataset
        self.calibration_data: List[CalibrationPoint] = []

        # Model weights (simple MLP: input → hidden → output)
        # Input: 20-dim (10 design + 10 workload features)
        # Output: 3-dim (latency_correction, energy_correction, area_correction)
        self.input_dim = 20
        self.output_dim = 3
        self._init_weights()

        # Running calibration statistics
        self.n_calibrations = 0
        self.mean_latency_error = 0.0
        self.mean_energy_error = 0.0

    def _init_weights(self):
        """Initialize simple MLP weights."""
        rng = np.random.default_rng(self.seed)
        scale = 0.01
        # Layer 1: input → hidden
        self.w1 = rng.normal(0, scale, (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        # Layer 2: hidden → output
        self.w2 = rng.normal(0, scale, (self.hidden_dim, self.output_dim)).astype(np.float32)
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    def encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """Encode accelerator design to feature vector."""
        features = np.zeros(10, dtype=np.float32)
        features[0] = np.log2(max(design.get("pe_array_rows", 16), 1)) / 7.0
        features[1] = np.log2(max(design.get("pe_array_cols", 16), 1)) / 7.0
        features[2] = np.log2(max(design.get("l0_buffer_kb", 1.0), 0.1) + 1) / 4.0
        features[3] = np.log2(max(design.get("l1_buffer_kb", 64.0), 1)) / 10.0
        features[4] = np.log2(max(design.get("l2_buffer_kb", 512.0), 1)) / 12.0

        # Categorical encodings
        dataflow_map = {"output_stationary": 0.25, "weight_stationary": 0.5, "row_stationary": 0.75}
        features[5] = dataflow_map.get(design.get("dataflow", ""), 0.0)

        noc_map = {"mesh": 0.33, "ring": 0.66, "tree": 1.0}
        features[6] = noc_map.get(design.get("noc_topology", ""), 0.0)

        prec_map = {"int8": 0.25, "fp16": 0.5, "bf16": 0.75, "fp32": 1.0}
        features[7] = prec_map.get(design.get("data_precision", ""), 0.0)

        features[8] = design.get("clock_mhz", 1000.0) / 2000.0
        features[9] = design.get("tech_node_nm", 7) / 28.0
        return features

    def encode_workload(self, workload: Optional[Dict[str, Any]]) -> np.ndarray:
        """Encode workload characteristics."""
        features = np.zeros(10, dtype=np.float32)
        if workload is None:
            return features
        features[0] = np.log10(max(workload.get("estimated_flops", 1e6), 1)) / 15.0
        features[1] = np.log10(max(workload.get("batch_size", 1), 1)) / 4.0
        features[2] = np.log10(max(workload.get("sequence_length", 1024), 1)) / 5.0
        features[3] = np.log10(max(workload.get("hidden_dim", 4096), 1)) / 5.0
        features[4] = workload.get("num_operators", 0) / 50.0
        return features

    def predict_correction(
        self,
        design: Dict[str, Any],
        workload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float]:
        """
        Predict correction factors for analytical model.

        Returns:
            (latency_correction, energy_correction, area_correction)
            Each is a multiplicative factor (1.0 = no correction)
        """
        x = np.concatenate([
            self.encode_design(design),
            self.encode_workload(workload),
        ])

        # Forward pass
        h = np.maximum(x @ self.w1 + self.b1, 0)  # ReLU
        out = h @ self.w2 + self.b2

        # Sigmoid to bound corrections to [0.5, 2.0] range
        corrections = 0.5 + 1.5 / (1.0 + np.exp(-out))

        return float(corrections[0]), float(corrections[1]), float(corrections[2])

    def add_calibration(self, point: CalibrationPoint):
        """Add a calibration data point and update model."""
        self.calibration_data.append(point)
        self.n_calibrations += 1

        # Compute errors
        if point.predicted_latency_ms > 0:
            point.latency_error_pct = abs(
                point.actual_latency_ms - point.predicted_latency_ms
            ) / point.predicted_latency_ms * 100

        if point.predicted_energy_pj > 0:
            point.energy_error_pct = abs(
                point.actual_energy_pj - point.predicted_energy_pj
            ) / point.predicted_energy_pj * 100

        # Online update (if we have enough data)
        if len(self.calibration_data) >= 10:
            self._update_model()

    def _update_model(self, epochs: int = 5):
        """Update model weights from calibration data."""
        if not self.calibration_data:
            return

        # Build training data
        X = []
        Y = []
        for point in self.calibration_data:
            x = np.concatenate([
                self.encode_design(point.design),
                self.encode_workload(point.workload),
            ])
            X.append(x)

            # Target: correction ratios
            lat_ratio = (
                point.actual_latency_ms / max(point.predicted_latency_ms, 1e-6)
                if point.predicted_latency_ms > 0 else 1.0
            )
            eng_ratio = (
                point.actual_energy_pj / max(point.predicted_energy_pj, 1e-6)
                if point.predicted_energy_pj > 0 else 1.0
            )
            area_ratio = (
                point.actual_area_mm2 / max(point.predicted_area_mm2, 1e-6)
                if point.predicted_area_mm2 > 0 else 1.0
            )
            Y.append([lat_ratio, eng_ratio, area_ratio])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Simple gradient descent
        for _ in range(epochs):
            # Forward
            h = np.maximum(X @ self.w1 + self.b1, 0)
            out = h @ self.w2 + self.b2
            pred = 0.5 + 1.5 / (1.0 + np.exp(-out))

            # Loss (MSE)
            error = pred - Y
            n = len(X)

            # Backward (simplified)
            sigmoid = (pred - 0.5) / 1.5
            d_out = error * 1.5 * sigmoid * (1 - sigmoid) / n

            # Update layer 2
            self.w2 -= self.learning_rate * (h.T @ d_out)
            self.b2 -= self.learning_rate * d_out.sum(axis=0)

            # Update layer 1
            d_h = d_out @ self.w2.T
            d_h[h <= 0] = 0  # ReLU mask
            self.w1 -= self.learning_rate * (X.T @ d_h)
            self.b1 -= self.learning_rate * d_h.sum(axis=0)

        # Update statistics
        if len(self.calibration_data) > 0:
            self.mean_latency_error = np.mean(
                [p.latency_error_pct for p in self.calibration_data]
            )
            self.mean_energy_error = np.mean(
                [p.energy_error_pct for p in self.calibration_data]
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "n_calibrations": self.n_calibrations,
            "mean_latency_error_pct": self.mean_latency_error,
            "mean_energy_error_pct": self.mean_energy_error,
            "calibration_data_size": len(self.calibration_data),
        }

    def save(self, path: str):
        """Save model weights and calibration data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(path / "weights.npz"),
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
        )

        cal_data = [p.to_dict() for p in self.calibration_data]
        with open(path / "calibration.json", "w") as f:
            json.dump(cal_data, f)

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """Load model weights and calibration data."""
        path = Path(path)
        model = cls()

        weights = np.load(str(path / "weights.npz"))
        model.w1 = weights["w1"]
        model.b1 = weights["b1"]
        model.w2 = weights["w2"]
        model.b2 = weights["b2"]

        cal_path = path / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal_data = json.load(f)
            for d in cal_data:
                valid_fields = CalibrationPoint.__dataclass_fields__
                filtered = {k: v for k, v in d.items() if k in valid_fields}
                model.calibration_data.append(CalibrationPoint(**filtered))
            model.n_calibrations = len(model.calibration_data)

        return model
