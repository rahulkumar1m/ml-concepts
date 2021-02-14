# Import the libraries we'll need
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, C=0.001, n_iters=100) -> None:
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert 0 labels to -1 to set threshold with hinge loss
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.rand(n_features)
        self.b = 0
    
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.C * self.w)
                else:
                    self.w -= self.lr * (2 * self.C * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        preds = np.dot(X, self.w) - self.b
        return np.sign(preds)
