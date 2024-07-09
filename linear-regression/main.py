

from linear_regression import LinearRegression, RidgeRegression
from encoder import OneHotEncoder

import numpy as np

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

def softmax(x):
    """
    turn vector into probability distribution
    """
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def generate(model, encoder, seed, length, do_softmax=False, separator=""):
    output = seed
    bias_term = np.array([1])
    current_str = seed.copy()
    idxs = np.array(list(encoder.obj_to_idx.values()))

    for _ in range(length):
        x_t = encoder.encode_cat(current_str, do_suffix=True)
        y = model.predict(x_t)
        y = y - np.min(y)
        y *= 10 / np.max(y)
        prob = softmax(y) if do_softmax else y / y.sum()
        # prob /= prob.sum() # fix rounding error
        next_idx = np.random.choice(idxs, p=prob)
        next_char = encoder.decode(next_idx)

        output += separator + next_char
        current_str = current_str[1:] + [next_char]
    
    return "".join(output)

def get_text():
    t = "Modern AI is pretty great. New systems are advancing at marvelous rates, improving year after year. Unfortunately, this technological progress has led to a very related, very human problem. Humans are treating AI like it's human. More specifically, humans are anthropomorphizing AI systems, attributing qualities like understanding, intentionality, and empathy to what are essentially glorified math equations. Anthropomorphizing AI can lead to misunderstandings about the true nature and limitations of these systems. By believing AI is knowledgeable and understanding, humans tend to overestimate AI's capabilities. They entrust data processing formulas with their emotional needs and nuanced ethical dilemmas, neither of which these tools are equipped to handle. This misplaced trust in algorithms can lead to bad decisions, but more importantly, it shifts blame away from the user and the programmer. ELIZA was an experiment conducted by Joseph Weizenbaum in 1966. ELIZA simulated a Rogerian therapist using very, very simple code. In fact, the bulk of the ELIZA algorithm took only 7 lines of code for me to replicate. Since Rogerian therapists primarily take what the patient says and turn it into a question, a simple procedure does a decent job for the simulation. Even though ELIZA understands nothing, participants often felt a close, empathetic connection with the program. Weizenbaum's own secretary allegedly asked Weizenbaum to leave the room when talking to ELIZA. Users told ELIZA their problems and feelings, projecting human traits onto the not-at-all-human program. ELIZA is a standout case when it comes to anthropomorphization. In fact, despite being orders of magnitude simpler, it outperformed GPT-3.5 in a Turing Test. To clarify, this program from the 1960s is better at pretending to be human than the 175 billion parameter behemoth from OpenAI. To underscore the absurdity of attributing true understanding to AI, I would like to introduce you to Anti-ELIZA. Anti-ELIZA starts conversations like ELIZA, using standard Rogerian therapy tactics. However, as the conversation progresses, Anti-ELIZA gradually shifts to reminding the user that ELIZA is just a few lines of code and incapable of understanding anything. By transitioning from seeming empathy to blunt reminders that the program isn’t conscious, Anti-ELIZA highlights the ease with which humans overestimate AI systems. Regardless of how convincing a conversation with AI like ELIZA may seem, the algorithm on the other side is devoid of sentience, comprehension, and emotion."
    t = t.lower()
    t = t.replace(".", "")
    t = t.replace(",", "")
    return t.split(" ")

def main():
    tuple_len = 1 # number of characters in each chunk, 1-10
    token_len = 1 # number of chunks per token, 1-5
    regularizer = 0.1 # prevents model overfitting. increasing regularizer smooths predictions, giving more variation in output, 0-0.1
    do_softmax = False # when True, makes most likely prediction even more likely
    num_generated_tokens = 1000 # how much to generate
    separator = " " # space for word level, empty string for character level

    text = get_text() # modify this for text input
    encoder, X, Y = tokenize(text, tuple_length=tuple_len, token_length=token_len)
    model = RidgeRegression()
    model.fit(X, Y, regularizer)
    print(generate(model, encoder, encoder.get_seed(token_len), num_generated_tokens, do_softmax=do_softmax, separator=separator))

if __name__ == "__main__":
    main()