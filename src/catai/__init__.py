"""Cashlog image classification package."""

__all__ = [
    "CashlogCategoryClassifier",
    "CashlogEnsembleClassifier",
    "load_cashlog_classifier_from_env",
]


def __getattr__(name: str):
    if name == "CashlogCategoryClassifier":
        from .cashlog_classifier import CashlogCategoryClassifier

        return CashlogCategoryClassifier
    if name == "CashlogEnsembleClassifier":
        from .cashlog_classifier import CashlogEnsembleClassifier

        return CashlogEnsembleClassifier
    if name == "load_cashlog_classifier_from_env":
        from .cashlog_classifier import load_cashlog_classifier_from_env

        return load_cashlog_classifier_from_env
    raise AttributeError(name)
