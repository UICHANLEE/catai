from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cashlog_classifier import CashlogCategoryClassifier, CashlogEnsembleClassifier


def predict_cashlog_main() -> None:
    parser = argparse.ArgumentParser(description="Predict Cashlog category from a product image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--ensemble-config", type=Path)
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.hybrid_config:
        from .cashlog_hybrid_classifier import CashlogHybridClassifier

        classifier = CashlogHybridClassifier.from_config(
            args.hybrid_config,
            device=args.device,
        )
    elif args.ensemble_config:
        classifier = CashlogEnsembleClassifier(
            ensemble_config=args.ensemble_config,
            device=args.device,
        )
    else:
        classifier = CashlogCategoryClassifier(
            checkpoint_path=args.checkpoint,
            labels_path=args.labels,
            device=args.device,
        )
    result = classifier.analyze(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    predict_cashlog_main()
