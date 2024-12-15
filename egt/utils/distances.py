"""
Authors: Ronik and Jon
"""


import numpy as np

def hamming(x, y):
    return int(sum(np.not_equal(x, y)))
