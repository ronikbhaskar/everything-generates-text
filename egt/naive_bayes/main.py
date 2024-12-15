
from naive_bayes import NaiveBayes
from egt.utils.one_hot_encoder import OneHotEncoder

import numpy as np

def generate(model, encoder, seed, num_tokens, separator=" "):
    output = seed
    
    current_str = seed

    for _ in range(num_tokens):
        x_t = encoder.encode(current_str)
        y = model.predict(x_t)

        output += separator + y
        current_str = y

    return output
        

def get_text():
    t = "Modern AI is pretty great. New systems are advancing at marvelous rates, improving year after year. Unfortunately, this technological progress has led to a very related, very human problem. Humans are treating AI like it's human. More specifically, humans are anthropomorphizing AI systems, attributing qualities like understanding, intentionality, and empathy to what are essentially glorified math equations. Anthropomorphizing AI can lead to misunderstandings about the true nature and limitations of these systems. By believing AI is knowledgeable and understanding, humans tend to overestimate AI's capabilities. They entrust data processing formulas with their emotional needs and nuanced ethical dilemmas, neither of which these tools are equipped to handle. This misplaced trust in algorithms can lead to bad decisions, but more importantly, it shifts blame away from the user and the programmer. ELIZA was an experiment conducted by Joseph Weizenbaum in 1966. ELIZA simulated a Rogerian therapist using very, very simple code. In fact, the bulk of the ELIZA algorithm took only 7 lines of code for me to replicate. Since Rogerian therapists primarily take what the patient says and turn it into a question, a simple procedure does a decent job for the simulation. Even though ELIZA understands nothing, participants often felt a close, empathetic connection with the program. Weizenbaum's own secretary allegedly asked Weizenbaum to leave the room when talking to ELIZA. Users told ELIZA their problems and feelings, projecting human traits onto the not-at-all-human program. ELIZA is a standout case when it comes to anthropomorphization. In fact, despite being orders of magnitude simpler, it outperformed GPT-3.5 in a Turing Test. To clarify, this program from the 1960s is better at pretending to be human than the 175 billion parameter behemoth from OpenAI. To underscore the absurdity of attributing true understanding to AI, I would like to introduce you to Anti-ELIZA. Anti-ELIZA starts conversations like ELIZA, using standard Rogerian therapy tactics. However, as the conversation progresses, Anti-ELIZA gradually shifts to reminding the user that ELIZA is just a few lines of code and incapable of understanding anything. By transitioning from seeming empathy to blunt reminders that the program isn’t conscious, Anti-ELIZA highlights the ease with which humans overestimate AI systems. Regardless of how convincing a conversation with AI like ELIZA may seem, the algorithm on the other side is devoid of sentience, comprehension, and emotion."
    t = t.lower()
    t = t.replace(".", "")
    t = t.replace(",", "")
    return t.split(" ")


def main():
    num_generated_tokens = 100 # how much to generate
    separator = " " # space for word level, empty string for character level

    text = get_text() # modify this for text input
    encoder = OneHotEncoder(text)
    X = np.array([encoder.encode(token) for token in text[:-1]])
    y = np.array(text[1:])
    model = NaiveBayes()
    model.fit(X, y)
    print(generate(model, encoder, encoder.get_seed(1)[0], num_generated_tokens, separator=separator))


if __name__ == "__main__":
    main()