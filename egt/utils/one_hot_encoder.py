

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
    
def tokenize(text, tuple_length=3, token_length=2):
    X = []
    Y = []
    tuples = []
    bias_term = np.array([1])

    if type(text) == list:
        tuples = [text[i:] for i in range(tuple_length)]
    else:
        for i in range(tuple_length):
            tuples.append([text[j:j+tuple_length] for j in range(i, len(text) - 1, tuple_length)])

    encoder = OneHotEncoder(sum(tuples, []), suffix=bias_term)

    for tuple_group in tuples:
        text_length = len(tuple_group)

        features = []
        labels = []

        for i in range(text_length - token_length):
            features.append(encoder.encode_cat(tuple_group[i:i+token_length], do_suffix=True))
            labels.append(encoder.encode(tuple_group[i + token_length]))

        X += features
        Y += labels

    return encoder, np.array(X), np.array(Y)