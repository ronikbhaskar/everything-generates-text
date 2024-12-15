
from collections import Counter
import numpy as np

def make_distribution(y):
    total = len(y)
    counter = Counter(y) 
    counts = counter.most_common()
    # first is values, second is probabilites
    return \
        np.array([pair[0] for pair in counts]), \
        np.array([pair[1] / total for pair in counts])

def entropy(y):
    probabilities = make_distribution(y)[1].astype(np.float16)
    return -np.sum(probabilities * np.log2(probabilities))