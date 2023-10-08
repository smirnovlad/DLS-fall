import numpy as np
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, fit_intercept=False):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, features, target):
        n, k = features.shape
        features_train = features
        if (self.fit_intercept):
            features_train = np.hstack((features_train, np.ones((n, 1))))
        self.w = np.linalg.inv(features_train.T @ features_train) @ features_train.T @ target

    def predict(self, features):
        n, k = features.shape
        if (self.fit_intercept):
            features = np.hstack((features, np.ones((n, 1))))
        return features @ self.w

    def get_weights(self):
        return self.w

class LinearRegressionGD(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, features, target, lr=0.02, steps=50):
        n, k = features.shape
        features_train = features
        if (self.fit_intercept):
            features_train = np.hstack((features_train, np.ones((n, 1))))

        self.w = np.zeros((k + self.fit_intercept, 1))
        self.losses = []
        for _ in range(steps):
            target_pred = features_train @ self.w
            self.losses.append(mean_squared_error(target, target_pred))
            grad_w = self._calc_gradient(features_train, target, target_pred)
            self.w -= lr * grad_w

    def _calc_gradient(self, features, target, target_pred):
        n = features.shape[0]
        return 2 * features.T @ (target_pred - target) / n

    def get_losses(self):
        return self.losses

class LinearRegressionSGD(LinearRegressionGD):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs)
        self.n_sample = n_sample

    def _calc_gradient(self, features, target, target_pred):
        n = features.shape[0]
        inds = np.random.choice(np.arange(n), size=self.n_sample, replace=False)
        return 2 * features[inds].T @ (target_pred[inds] - target[inds]) / self.n_sample