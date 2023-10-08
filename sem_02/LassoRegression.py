import numpy as np
from LinearRegression import LinearRegression

def soft_sign(x, eps=1e-7):
    if (abs(x) > eps):
        return np.sign(x)
    return x / eps

np_soft_sign = np.vectorize(soft_sign)

class LASSORegression(LinearRegression):
    def __init__(self, alpha=1.0, lr=0.02, steps=100, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lr = lr
        self.steps = steps
        self.n_sample = n_sample

    def fit(self, features, target):
        n, k = features.shape
        if (self.fit_intercept):
            features = np.hstack((features, np.ones((n, 1))))
        self.w = np.zeros((features.shape[1], 1))
        for _ in range(self.steps):
            target_pred = features @ self.w
            grad_w = self._calc_gradient(features, target, target_pred)
            self.w -= self.lr * grad_w
    def _calc_gradient(self, features, target, target_pred):
        n = features.shape[0]
        return features.T @ (target_pred - target) / n + self.alpha * np_soft_sign(self.w)
