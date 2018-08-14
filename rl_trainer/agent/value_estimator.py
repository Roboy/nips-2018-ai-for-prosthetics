from typing import Sequence


class ValueEstimator:
    """Based on sklearn BaseEstimator"""

    def predict(self, X) -> Sequence[float]:
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError


class MockEstimator(ValueEstimator):

    def predict(self, X) -> Sequence[float]:
        return [0.5] * len(X)

    def fit(self, X, y):
        pass
