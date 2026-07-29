"""Small probability-level ensembles used by CashLog model artifacts."""

from __future__ import annotations

import numpy as np


class ProbabilityBlendClassifier:
    """Blend two classifiers while aligning their class probability columns."""

    def __init__(self, baseline, candidate, candidate_weight: float):
        if not 0.0 <= candidate_weight <= 1.0:
            raise ValueError("candidate_weight must be between 0 and 1")
        self.baseline = baseline
        self.candidate = candidate
        self.candidate_weight = float(candidate_weight)
        self.classes_ = np.asarray(
            sorted({str(value) for value in baseline.classes_} | {str(value) for value in candidate.classes_})
        )

    def _aligned_probabilities(self, model, features: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(features)
        aligned = np.zeros((len(features), len(self.classes_)), dtype=probabilities.dtype)
        output_indexes = {str(value): index for index, value in enumerate(self.classes_)}
        for source_index, class_id in enumerate(model.classes_):
            aligned[:, output_indexes[str(class_id)]] = probabilities[:, source_index]
        return aligned

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        baseline = self._aligned_probabilities(self.baseline, features)
        candidate = self._aligned_probabilities(self.candidate, features)
        return (1.0 - self.candidate_weight) * baseline + self.candidate_weight * candidate

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(features).argmax(axis=1)]
