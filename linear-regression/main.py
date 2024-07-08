

from linear_regression import LinearRegression, RidgeRegression
from encoder import OneHotEncoder

import numpy as np

def encode_concat(encoder, objs, suffix=None):
    return np.concat([encoder.encode(obj) for obj in objs] + [suffix])

def tokenize(text, encoder, token_length=4):
    X = []
    Y = []
    text_length = len(text)
    bias_term = np.array([1])

    enc_cat = lambda objs: encode_concat(encoder, objs, suffix=bias_term)

    for i in range(text_length - token_length):
        X.append(enc_cat(text[i:i+token_length]))
        Y.append(encoder.encode(text[i + token_length]))

    return np.array(X), np.array(Y)

def softmax(x):
    """
    turn vector into probability distribution
    """
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def generate(model, encoder, seed, length, do_softmax=False):
    output = seed
    bias_term = np.array([1])
    enc_cat = lambda objs: encode_concat(encoder, objs, suffix=bias_term)
    current_str = seed.copy()
    idxs = np.array(list(encoder.obj_to_idx.values()))

    for _ in range(length):
        x_t = enc_cat(current_str)
        y = model.predict(x_t)
        y = y - np.min(y)
        y *= 10 / np.max(y)
        prob = softmax(y) if do_softmax else y / y.sum()
        # prob /= prob.sum() # fix rounding error
        next_idx = np.random.choice(idxs, p=prob)
        next_char = encoder.decode(next_idx)

        output += next_char
        current_str = current_str[1:] + [next_char]
    
    return "".join(output)

def get_text():
    return "Modern AI is pretty great. New systems are advancing at marvelous rates, improving year after year. Unfortunately, this technological progress has led to a very related, very human problem. Humans are treating AI like it's human. More specifically, humans are anthropomorphizing AI systems, attributing qualities like understanding, intentionality, and empathy to what are essentially glorified math equations. Anthropomorphizing AI can lead to misunderstandings about the true nature and limitations of these systems. By believing AI is knowledgeable and understanding, humans tend to overestimate AI's capabilities. They entrust data processing formulas with their emotional needs and nuanced ethical dilemmas, neither of which these tools are equipped to handle. This misplaced trust in algorithms can lead to bad decisions, but more importantly, it shifts blame away from the user and the programmer."

def main():
    tuple_len = 3 # number of characters in each chunk, 1-5
    token_len = 2 # number of chunks per token, 1-5
    regularizer = 0.001 # prevents model overfitting. increasing regularizer smooths predictions, giving more variation in output, 0-0.1
    do_softmax = False # when True, makes most likely prediction even more likely
    num_generated_tokens = 100 # how much to generate

    text = get_text() # modify this for text input
    tuples = [text[i:i+tuple_len] for i in range(0, len(text) - 1, tuple_len)]
    encoder = OneHotEncoder(tuples)
    X, Y = tokenize(tuples, encoder, token_length=token_len)
    model = RidgeRegression()
    model.fit(X, Y, regularizer)
    print(generate(model, encoder, tuples[:token_len], num_generated_tokens, do_softmax=do_softmax))

if __name__ == "__main__":
    main()