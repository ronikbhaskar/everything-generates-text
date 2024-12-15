"""
Remember when I said linear regression was gonna be weird.
This is gonna be weirder.
"""

import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, iters=1000):
        """
        lr and iters in case of gradient-based optimization approach
        """
        self.lr = lr
        self.iters = iters
        self.W = None
        self.b = None

    def fit_analytical(self, X, Y, lam):
        """
        need to fit multiple logistic regressions
        each categorical variable (token) is reduced to a binary variable,
        
        Y = sigmoid(X W)
        inv_sigmoid(Y) = X W
        W = (X.T X)^-1 X.T inv_sigmoid(Y)

        in practice: Y is adjusted so 0 -> 0.00001 and 1 -> 0.99999
        """
        
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)
        self.lam = lam

        num_features = self.X_train.shape[1]
        I = np.eye(num_features)
        lam_I = self.lam * I

        # element-wise operations
        self.Y_train = self.Y_train * 0.99998 + 0.00001
        inv_sig_Y_train = np.log(self.Y_train / (1 - self.Y_train)) 

        self.W = np.linalg.inv(self.X_train.T @ self.X_train + lam_I) @ \
                       self.X_train.T @ inv_sig_Y_train

    def predict(self, x):
        """
        Y = sigmoid(x @ W)
        """

        return 1 / (1 + np.exp(-x @ self.W))