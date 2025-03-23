

import numpy as np
import re

class OneHotEncoder:

    def __init__(self, collection, suffix=None):
        """
        collection : str | list | set
        """

        self.collection = set(collection)
        self.length = len(self.collection)
        self.idx_to_obj = list(self.collection)
        self.obj_to_idx = {obj:i for i, obj in enumerate(self.idx_to_obj)}
        self.idxs = np.array(list(self.obj_to_idx.values()))
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
    
    def sample_from_probs(self, probs):
        """
        samples and returns object using probabilities as weights
        """

        return self.decode(np.random.choice(self.idxs, p=probs))
    
    def get_seed(self, idx=0):
        """
        """

        return self.idx_to_obj[idx]
    
def embed(text, tuple_length=1):
    X = []
    Y = []
    tuples = []
    bias_term = np.array([1])

    tuples = [text[i:] for i in range(tuple_length)]

    encoder = OneHotEncoder(sum(tuples, []), suffix=bias_term)

    for tuple_group in tuples:
        text_length = len(tuple_group)

        features = []
        labels = []

        for i in range(text_length - 1):
            features.append(encoder.encode_cat(tuple_group[i:i+1], do_suffix=True))
            labels.append(encoder.encode(tuple_group[i + 1]))

        X += features
        Y += labels

    return encoder, np.array(X), np.array(Y)

def generate(model, encoder, seed=None, length=100, separator=" ", do_suffix=False):
    if seed == None:
        seed = encoder.get_seed()
    
    output = seed.capitalize()
    current_str = seed

    was_period = current_str == ".";

    for _ in range(length):
        x_t = encoder.encode(current_str, do_suffix=do_suffix)
        y = model.predict(x_t)
        if isinstance(y, np.ndarray):
            next_token = encoder.sample_from_probs(y)
        else:
            next_token = y

        if not (next_token == "." or next_token == ","):
            output += separator

        if was_period:
            output += next_token.capitalize()
        else:
            output += next_token

        current_str = next_token
        was_period = current_str == ".";
    
    return output

def tokenize(text):
    """Tokenize, as back-translated from my markov site."""
    # End of sentence punctuation
    text = re.sub(r'[\.\?!;]+', ' . ', text)
    # End of phrase punctuation
    text = re.sub(r'--|[,:()]+', ' , ', text)
    # Remove quotes
    text = re.sub(r'"', '', text)
    text = re.sub(r'( \')|(\' )', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess spaces
    text = re.sub(r'[ \t\n]+', ' ', text)
    # Capitalization of 'I'
    text = re.sub(r' i ', ' I ', text)
    text = re.sub(r' i\'', ' I\'', text)
    
    # Tokenize by splitting on spaces and filter out empty tokens
    text_list = [word for word in text.split(' ') if word]
    
    return text_list