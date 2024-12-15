

import numpy as np

class OneHotEncoder:

    def __init__(self, collection, suffix=None):
        """
        collection : str | list | set
        """

        self.collection = set(collection)
        self.length = len(self.collection)
        self.idx_to_obj = list(self.collection)
        self.obj_to_idx = {obj:i for i, obj in enumerate(self.idx_to_obj)}
        if suffix:
            self.suffix = suffix
        else:
            self.suffix = np.array([])

        self.one_hot_vectors = np.eye(self.length)

    def _add_suffix(self, obj):
        return np.concat([obj, self.suffix])

    def encode(self, obj, do_suffix=False):
        """
        converts object to one-hot vector
        """
        oh = self.one_hot_vectors[self.obj_to_idx[obj]]
        if do_suffix:
            return self._add_suffix(oh)
        return oh
    
    def encode_cat(self, objs, do_suffix=False):
        """
        encode multiple objects and concatenate them
        """
        oh = np.concat([self.one_hot_vectors[self.obj_to_idx[obj]] for obj in objs])
        if do_suffix:
            return self._add_suffix(oh)
        return oh

    def decode(self, idx):
        """
        converts index to object
        """

        return self.idx_to_obj[idx]
    
    def get_seed(self, length):
        """
        """

        return self.idx_to_obj[:length]