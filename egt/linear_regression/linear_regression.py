"""
Oh boy, this is gonna be weird.
"""

import numpy as np

class LinearRegression:

    def __init__(self):
        pass

    def fit(self, X, Y):
        """
        X is n by m
        Y is n by k

        Y = X W
        W = (X.T X)^-1 X.T Y
        
        Essentially, this just performs k linear regressions
        on the same training data for k different label sets
        all at once
        """
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)

        self.W = np.linalg.inv(self.X_train.T @ self.X_train) @ \
                 self.X_train.T @ self.Y_train
        
    def predict(self, x):
        """
        simple linear operation
        sike, I have to revamp this to do a weighted selection of the next token
        so I can create a universal generate function for the OneHotEncoder
        """

        y = x @ self.W
        y = y - np.min(y)
        y *= 10 / np.max(y)
        return y / y.sum() # probability vector
    
class RidgeRegression(LinearRegression):
    """
    aka L2 regularization
    this reduces overfitting, which I fear for the linear models
    """

    def fit(self, X, Y, lam):
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)
        self.lam = lam

        num_features = self.X_train.shape[1]
        I = np.eye(num_features)
        lam_I = self.lam * I

        self.W = np.linalg.inv(self.X_train.T @ self.X_train + lam_I) @ \
                 self.X_train.T @ self.Y_train   
    