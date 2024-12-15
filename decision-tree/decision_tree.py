
import numpy as np
import random
from collections import Counter

from mcnf import MCNF

class Node:

    def __init__(
        self,
        mcnf=None,
        threshold=None,
        left=None,
        right=None,
        values=None,
        probs=None
    ):
        self.mcnf = mcnf
        self.threshold = threshold
        self.left = left
        self.right = right
        self.values = values
        self.probs = probs

    def is_leaf_node(self):
        return self.values is not None

def _make_distribution(y):
    total = len(y)
    counter = Counter(y) 
    counts = counter.most_common()
    # first is values, second is probabilites
    return \
        np.array([pair[0] for pair in counts]), \
        np.array([pair[1] / total for pair in counts]) 

def _entropy(y):
    probabilities = _make_distribution(y)[1].astype(np.float16)
    return -np.sum(probabilities * np.log2(probabilities))
    
class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        n_unique_samples = len(np.unique(X, axis=0))

        assert not (n_labels < 1), "nothing to predict here, whatcha tryna do?"

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_unique_samples == 1 or n_samples < self.min_samples_split:
            leaf_values, leaf_probs = _make_distribution(y)
            return Node(values=leaf_values, probs=leaf_probs)

        # find the best formula
        best_formula, best_threshold = self._greedy_best_formula(X, y)

        if best_threshold == None:
            leaf_values, leaf_probs = _make_distribution(y)
            return Node(values=leaf_values, probs=leaf_probs)

        # create child nodes
        best_feature = np.array([best_formula.evaluate(row) for row in X])
        left_idxs = np.argwhere(best_feature <= best_threshold).flatten()
        right_idxs = np.argwhere(best_feature > best_threshold).flatten()
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_formula, best_threshold, left, right)

    def _greedy_best_formula(self, X, y):
        """
        not provably optimal, but certainly faster than full search
        """

        base_entropy = _entropy(y)
        num_samples = len(y)

        best_formula, best_threshold = MCNF(X.shape[1]), None

        best_gain = 0

        while 1:
            temp_best_formula = best_formula
            temp_best_threshold = best_threshold
            temp_best_info_gain = best_gain
            # print(temp_best_formula.clause_sets)
            for i, column in enumerate(X.T):
                values = np.unique(column)
                # print(f"values: {values}")
                for value in values:
                    # only evaluate things we haven't used in the formula
                    if value in best_formula.clause_sets[i]:
                        continue

                    temp_formula = best_formula.update(value, i)
                    new_feature = np.array([temp_formula.evaluate(row) for row in X])

                    thresholds = np.unique(new_feature)

                    for threshold in thresholds:
                        left_idxs = np.argwhere(new_feature <= threshold).flatten()
                        right_idxs = np.argwhere(new_feature > threshold).flatten()
                        
                        num_left, num_right = len(left_idxs), len(right_idxs)

                        if num_left == 0 or num_right == 0:
                            info_gain = 0
                        else:
                            info_gain = base_entropy - (num_left * _entropy(y[left_idxs]) + num_right * _entropy(y[right_idxs])) / num_samples

                        if info_gain > temp_best_info_gain:
                            temp_best_info_gain = info_gain
                            temp_best_formula = temp_formula
                            temp_best_threshold = threshold
                    

            if temp_best_info_gain <= best_gain:
                # we haven't improve the decision making, just stop
                break
            else:
                # we're still improving
                best_formula = temp_best_formula
                best_threshold = temp_best_threshold
                best_gain = temp_best_info_gain

        return best_formula, best_threshold


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calculate the information gain
                gain = None

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def predict(self, x):
        return self._traverse_tree(x, self.root)

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return np.random.choice(node.values, size=1, p=node.probs)[0]
        
        if node.mcnf.evaluate(x) <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
