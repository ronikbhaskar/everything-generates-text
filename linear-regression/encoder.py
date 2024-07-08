

import numpy as np

class OneHotEncoder:

    def __init__(self, collection):
        """
        collection : str | list | set
        """

        self.collection = set(collection)
        self.length = len(self.collection)
        self.idx_to_obj = list(self.collection)
        self.obj_to_idx = {obj:i for i, obj in enumerate(self.idx_to_obj)}

        self.one_hot_vectors = np.eye(self.length)

    def encode(self, obj):
        """
        converts object to one-hot vector
        """
        return self.one_hot_vectors[self.obj_to_idx[obj]]
    
    def decode(self, idx):
        """
        converts index to object
        """

        return self.idx_to_obj[idx]