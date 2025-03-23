

from linear_regression import LinearRegression, RidgeRegression
from egt.utils.one_hot_encoder import embed, generate
from egt.utils.example import get_text

import numpy as np

def main():
    tuple_len = 1 # number of characters in each chunk, 1-10
    regularizer = 0.01 # prevents model overfitting. increasing regularizer smooths predictions, giving more variation in output, 0-0.1
    num_generated_tokens = 100 # how much to generate
    separator = " " # space for word level, empty string for character level

    text = get_text() # modify this for text input
    encoder, X, Y = embed(text, tuple_length=tuple_len)
    model = RidgeRegression()
    model.fit(X, Y, regularizer)
    print(generate(model, encoder, encoder.get_seed(), num_generated_tokens, separator=separator, do_suffix=True))

if __name__ == "__main__":
    main()