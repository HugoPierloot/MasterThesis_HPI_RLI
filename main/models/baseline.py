# ─────────────────────────────────────────────────────────────
#  Baseline
#
#  Naive baseline classifiers.
#  Purpose: establish the performance floor that every real model  must beat. If a model can't beat the baseline, it adds no value.
#
#  Baselines implemented:
#    1. MajorityClassifier  — always predicts 1 (Success)
#       Rationale: represents the "accuracy trap" for imbalanced data.
#       A naive observer who says "every launch succeeds" gets 91.2%
#       accuracy. Any real model must beat this on F1/recall/AUC.
#
#    2. PriorRateClassifier — predicts 1 with probability = train success rate
#       Useful as a probabilistic baseline for AUC-ROC comparison.
# ─────────────────────────────────────────────────────────────

# Imports
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """
    Always predicts the majority class (1 = Success).
    Sklearn-compatible works with evaluate_model().
    """

    def fit(self, X, y):
        self.majority_class_ = int(np.bincount(y).argmax())
        self.classes_        = np.array([0, 1])
        self.prior_          = np.mean(y == 1)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        """Returns constant probability equal to the training success rate."""
        check_is_fitted(self)
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1 - self.prior_
        proba[:, 1] = self.prior_
        return proba


class PriorRateClassifier(BaseEstimator, ClassifierMixin):
    """
    Predicts 1 with probability = training set success rate.
    Useful as a random probabilistic baseline for AUC comparison.
    """

    def fit(self, X, y):
        self.prior_   = np.mean(y == 1)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        check_is_fitted(self)
        rng   = np.random.default_rng(42)
        preds = (rng.random(len(X)) < self.prior_).astype(int)
        return preds

    def predict_proba(self, X):
        check_is_fitted(self)
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1 - self.prior_
        proba[:, 1] = self.prior_
        return proba