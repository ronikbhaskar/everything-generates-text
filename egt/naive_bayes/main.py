
from naive_bayes import NaiveBayes
from egt.utils.one_hot_encoder import OneHotEncoder, generate, tokenize
from egt.utils.example import get_text

import numpy as np

def main():
    num_generated_tokens = 100 # how much to generate
    separator = " " # space for word level, empty string for character level
    text = get_text() # modify this for text input
    encoder = OneHotEncoder(text)
    X = np.array([encoder.encode(token) for token in text[:-1]])
    y = np.array(text[1:])
    model = NaiveBayes()
    model.fit(X, y)
    print(generate(model, encoder, encoder.get_seed(1), num_generated_tokens, separator=separator))


if __name__ == "__main__":
    main()