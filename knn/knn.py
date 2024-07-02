"""
Authors: Ronik and Jon
"""


import numpy as np
from random import choice

class KNN:

    def __init__(self, k, distance):
        assert k >= 1 and int(k) == k
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        """
        This method allow for ties for each position, 
        so there could be 2 nearest, 3 second-nearest, 1 third-nearest, etc
        until you get to the k-nearest.

        Uses random selection from these labels.
        """

        dists = [self.distance(x, x_t) for x_t in self.X_train]
        sorted_dists_unique = list(sorted(set(dists)))
        max_dist = sorted_dists_unique[self.k - 1]

        num_viable = sum(1 if d <= max_dist else 0 for d in dists)
        k_neighbor_addresses = np.argsort(dists)[:num_viable]
        k_neighbors = [self.y_train[i] for i in k_neighbor_addresses]
        
        return choice(k_neighbors)
