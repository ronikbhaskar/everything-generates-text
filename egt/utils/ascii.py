
import numpy as np

def tokenize(text, token_length=5):
    X = []
    y = []
    text_length = len(text)
    ascii_seq = list(map(ord, text))

    for i in range(text_length - token_length):
        X.append(ascii_seq[i:i+token_length])
        y.append(ascii_seq[i + token_length])

    return np.array(X), np.array(y)

def generate(model, seed, length):
    output = seed
    x = list(map(ord, seed))

    for i in range(length):
        next_char = model.predict(x)
        x[0] = next_char
        x = np.roll(x, -1)
        output += chr(next_char)

    return output