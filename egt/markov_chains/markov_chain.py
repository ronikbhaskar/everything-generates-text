
import numpy as np
from collections import Counter


class MarkovChain:
    """
    god, I've written markov chain text generators so many times that at this point, it's not even funny
    
    I suppose this is the first time I get to use numpy and Counter, though
    """

    def __init__(self):
        self.table = dict()

    def fit(self, X):
        tokens = np.unique(X[:, 0])
        for token in tokens:
            y_tokens = X[:, 1][X[:, 0] == token]
            total = len(y_tokens)
            counts = Counter(y_tokens).most_common()
            self.table[token] = [
                np.array([pair[0] for pair in counts]), # values
                np.array([pair[1] / total for pair in counts]) # probabilities
            ]

    def predict(self, x):
        entry = self.table[x]
        return np.random.choice(entry[0], size=1, p=entry[1])[0]
