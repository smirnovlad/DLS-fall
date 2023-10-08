import numpy as np
from LinearRegression import LinearRegression


class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, features, target):
        n, k = features.shape
        if (self.fit_intercept):
            features = np.concatenate((features, np.ones((n, 1))), axis=1)
        # lambdaI = self.alpha * np.eye(k)
        # if (self.fit_intercept):
        #     lambdaI = np.hstack((lambdaI, np.zeros((k, 1))))
        #     lambdaI = np.vstack((lambdaI, np.zeros((1, k + 1))))
        lambdaI = self.alpha * np.eye(features.shape[1])
        if (self.fit_intercept):
            lambdaI[-1, -1] = 0
        self.w = np.linalg.inv(features.T @ features + lambdaI) @ features.T @ target

class RidgeRegressionSGD(RidgeRegression):
    def __init__(self, lr=0.02, steps=100, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.steps = steps
        self.n_sample = n_sample

    def fit(self, features, target):
        n, k = features.shape
        if (self.fit_intercept):
            features = np.concatenate((features, np.ones((n, 1))), axis=1)
        self.w = np.zeros((features.shape[1], 1))
        for _ in range(self.steps):
            grad_w = self._calc_gradient(features, target)
            self.w -= self.lr * grad_w

    def _calc_gradient(self, features, target):
        lambdaI = self.alpha * np.eye(features.shape[1])
        inds = np.random.choice(features.shape[0], size=self.n_sample, replace=False)
        if (self.fit_intercept):
            lambdaI[-1, -1] = 0
        return 2 * (features[inds].T @ features[inds] + lambdaI) @ self.w - 2 * features[inds].T @ target[inds
        ]