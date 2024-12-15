"""
Author: Ronik and Jon
"""

#comment

from egt.utils.distances import hamming
from egt.utils.ascii import tokenize, generate
from knn import KNN

import numpy as np

def main():
    text = "Modern AI is pretty great. New systems are advancing at marvelous rates, improving year after year. Unfortunately, this technological progress has led to a very related, very human problem. Humans are treating AI like it's human. More specifically, humans are anthropomorphizing AI systems, attributing qualities like understanding, intentionality, and empathy to what are essentially glorified math equations. Anthropomorphizing AI can lead to misunderstandings about the true nature and limitations of these systems. By believing AI is knowledgeable and understanding, humans tend to overestimate AI's capabilities. They entrust data processing formulas with their emotional needs and nuanced ethical dilemmas, neither of which these tools are equipped to handle. This misplaced trust in algorithms can lead to bad decisions, but more importantly, it shifts blame away from the user and the programmer."
    X, y = tokenize(text, token_length=5)
    model = KNN(k=1, distance=hamming)
    model.fit(X, y)
    output = generate(model, "Moder", 500)
    print(output)

if __name__ == "__main__":
    main()