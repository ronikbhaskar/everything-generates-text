
import numpy as np

from markov_chain import MarkovChain
from egt.utils.index_encoder import IndexEncoder, generate
from egt.utils.example import get_text

def main():
    text = get_text()
    encoder = IndexEncoder(text)
    x = np.array([encoder.encode(token) for token in text])
    X = np.array([x, np.roll(x, -1)]).T
    model = MarkovChain()
    model.fit(X)

    print(generate(model, encoder))
    

if __name__ == "__main__":
    main()