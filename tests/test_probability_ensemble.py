import numpy as np

from catai.probability_ensemble import ProbabilityBlendClassifier


class StubClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        return np.repeat(self.probabilities[None, :], len(features), axis=0)


def test_probability_blend_aligns_class_columns():
    baseline = StubClassifier(["a", "b"], [0.8, 0.2])
    candidate = StubClassifier(["b", "c"], [0.6, 0.4])
    model = ProbabilityBlendClassifier(baseline, candidate, 0.25)

    result = model.predict_proba(np.zeros((1, 2)))

    assert model.classes_.tolist() == ["a", "b", "c"]
    assert np.allclose(result, [[0.6, 0.3, 0.1]])
