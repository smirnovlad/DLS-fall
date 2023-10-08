import numpy as np
from sklearn.metrics import mean_squared_error

import LinearRegression


class LogisticRegression:
    def __init__(self, fit_intercept=False):
        self.w = None
        self.fit_intercept=fit_intercept

    def fit(self, features, target, lr=0.02, max_iter=100):
        n, k = features.shape
        if (self.fit_intercept):
            features = np.concatenate((np.ones((n, 1)), features), axis=1)
        self.w = np.ones((k + 1 if self.fit_intercept else k, 1))
        self.losses = []

        for _ in range(max_iter):
            target_pred = self._sigmoid(features @ self.w)
            self.losses.append(mean_squared_error(target, target_pred))
            grad_w = self._calc_gradient(target_pred, target, features)
            self.w -= lr * grad_w.T

    def predict(self, features, threshold=0.5):
        return self.predict_proba(features) >= threshold

    def predict_proba(self, features):
        n = features.shape[0]
        if (self.fit_intercept):
            features = np.concatenate((np.ones((n, 1)), features), axis=1)
        return self._sigmoid(features @ self.w)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calc_gradient(self, target_pred, target, features):
        return (target_pred - target).T @ features