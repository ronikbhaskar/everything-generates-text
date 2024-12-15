
from egt.decision_tree.decision_tree import DecisionTree
from egt.utils.probability import make_distribution

import numpy as np

def _bootstrap_samples(X, y, n_trees):
    n_samples = X.shape[0]
    # boostrap sampling for standard random forests is too good at preventing overfitting
    # however, I need lots of overfitting to make these models work
    temp = 0.1
    idxs = np.random.choice(list(range(n_samples)) * 2, int((2 - temp) * n_samples), replace=False)
    return X[idxs], y[idxs]

class RandomForest:

    def __init__(
        self, 
        n_trees=10, 
        max_depth=10, 
        min_samples_split=2, 
        n_features=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.num_features=n_features
        self.trees=[]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = _bootstrap_samples(X, y, self.n_trees)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        conf_preds = [tree.predict(x, get_conf=True) for tree in self.trees]
        values, probs = np.array([pred[0] for pred in conf_preds]), np.array([pred[1] for pred in conf_preds])
        temp = 1
        probs = np.exp(probs / temp)
        probs /= np.sum(probs)
        # print(values, probs)
        # values, probs = make_distribution(values)

        return np.random.choice(values, size=1, p=probs)[0]