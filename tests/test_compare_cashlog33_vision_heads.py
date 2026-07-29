import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.compare_cashlog33_vision_heads import evaluate


class CompareCashlogVisionHeadsTests(unittest.TestCase):
    def test_evaluate_returns_metrics_and_per_leaf_recall(self) -> None:
        embeddings = np.asarray([[0.0], [0.1], [0.9], [1.0]])
        labels = ["left", "left", "right", "right"]
        model = LogisticRegression(random_state=7).fit(embeddings, labels)
        metrics = evaluate(model, embeddings, labels)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(set(metrics["per_leaf_recall"]), {"left", "right"})
        self.assertGreaterEqual(metrics["top3_accuracy"], metrics["top1_accuracy"])


if __name__ == "__main__":
    unittest.main()
