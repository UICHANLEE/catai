"""Cashlog image classification package."""

__all__ = ["CashlogCategoryClassifier"]


def __getattr__(name: str):
    if name == "CashlogCategoryClassifier":
        from .cashlog_classifier import CashlogCategoryClassifier

        return CashlogCategoryClassifier
    raise AttributeError(name)
