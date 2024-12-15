

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
    t = "Modern AI is pretty great. New systems are advancing at marvelous rates, improving year after year. Unfortunately, this technological progress has led to a very related, very human problem. Humans are treating AI like it's human. More specifically, humans are anthropomorphizing AI systems, attributing qualities like understanding, intentionality, and empathy to what are essentially glorified math equations. Anthropomorphizing AI can lead to misunderstandings about the true nature and limitations of these systems. By believing AI is knowledgeable and understanding, humans tend to overestimate AI's capabilities. They entrust data processing formulas with their emotional needs and nuanced ethical dilemmas, neither of which these tools are equipped to handle. This misplaced trust in algorithms can lead to bad decisions, but more importantly, it shifts blame away from the user and the programmer. ELIZA was an experiment conducted by Joseph Weizenbaum in 1966. ELIZA simulated a Rogerian therapist using very, very simple code. In fact, the bulk of the ELIZA algorithm took only 7 lines of code for me to replicate. Since Rogerian therapists primarily take what the patient says and turn it into a question, a simple procedure does a decent job for the simulation. Even though ELIZA understands nothing, participants often felt a close, empathetic connection with the program. Weizenbaum's own secretary allegedly asked Weizenbaum to leave the room when talking to ELIZA. Users told ELIZA their problems and feelings, projecting human traits onto the not-at-all-human program. ELIZA is a standout case when it comes to anthropomorphization. In fact, despite being orders of magnitude simpler, it outperformed GPT-3.5 in a Turing Test. To clarify, this program from the 1960s is better at pretending to be human than the 175 billion parameter behemoth from OpenAI. To underscore the absurdity of attributing true understanding to AI, I would like to introduce you to Anti-ELIZA. Anti-ELIZA starts conversations like ELIZA, using standard Rogerian therapy tactics. However, as the conversation progresses, Anti-ELIZA gradually shifts to reminding the user that ELIZA is just a few lines of code and incapable of understanding anything. By transitioning from seeming empathy to blunt reminders that the program isn't conscious, Anti-ELIZA highlights the ease with which humans overestimate AI systems. Regardless of how convincing a conversation with AI like ELIZA may seem, the algorithm on the other side is devoid of sentience, comprehension, and emotion."
    t = """This course is an introduction to key mathematical concepts at the heart of machine learning. The focus is on matrix methods and statistical models and features real-world applications ranging from classification and clustering to denoising and recommender systems. Mathematical topics covered include linear equations, regression, regularization, the singular value decomposition, iterative optimization algorithms, and probabilistic models. Machine learning topics include classification and regression, overfitting, support vector machines, kernel methods, clustering, neural networks, and deep learning. Students are expected to have taken a course in calculus and have exposure to numerical computing (e.g. Matlab, Python, Julia, or R). Knowledge of linear algebra and statistics is not assumed. Appropriate for graduate students or advanced undergraduates. This course could be used as a precursor to TTIC 31020, “Introduction to Machine Learning” or CSMC 35400. Recorded videos and lecture notes from previous years are available here. However, some content may be different this year. You are responsible for the material covered in class. We will be using Ed Discussion for announcements and class discussion. Using Ed will ensure that you get fast and efficient help from classmates, the TAs, and instructors. Rather than emailing questions to the teaching staff, please post your questions on Ed Discussion. Individual emails will not receive a response. You are responsible for keeping up with announcements posted on Ed by instructors, including clarifications, deadlines and forms for alerting staff to course conflicts with exam times, SDS accommodations, etc. Students are expected to have taken a course in calculus and have exposure to numerical computing (e.g. Matlab, Python, Julia, or R). All students will be evaluated by regular homework assignments and exams, with students in CMSC 35300 also completing a final project. One of our goals with the homework and exams is to assess to what extent you have mastered the material covered in the class. If you are using built-in packages for computing least squares (or other packages for things outside the scope of this class, like random forests, etc.), it does not demonstrate that you understand that is being taught in this course. No points will be awarded for solutions of that kind. We provide multiple demos that illustrate our expectations for coded solutions. Students are very welcome to work together on homework, but everyone needs to submit their own write-ups and solutions. To be completely clear, this means that you may NOT copy each other's solutions verbatim. Again, verbatim copies do not demonstrate your mastery of the material in the course. Your lowest homework score will not be counted towards your final grade. This policy allows you to miss an assignment, but only one. This should be used to cover illness, family emergencies, job interviews, or any other extenuating circumstance, but no more than once. Homework will be due on Sundays, and solutions posted on Wednesdays. Late homework will lose 10% of the available points per day late, and no credit after solutions are posted. A grade of P is given only for work of C quality or higher. You must request Pass/Fail grading prior to the day of the last lecture. All course materials are copyrighted. Posting any course materials online, including on CourseHero, is a violation of course policy and a breach of academic integrity. Students who violate this policy will be subject to disciplinary action. In this course, we will be developing skills and knowledge that are important to discover and practice on your own. Because use of AI tools inhibits development of these skills and knowledge, students are not allowed to use any AI tools, such as ChatGPT or Dall-E 2, in this course. Students are expected to present work that is their own without assistance from including automated tools. If you are unclear if something is an AI tool, please check with your instructor. Using AI tools for any purposes in this course will violate the University's academic integrity policy. Students who engage in academic misconduct or otherwise violate the standards of the University community may be brought before the College's Area Disciplinary Committee, or referred (as appropriate) to either the University-wide Disciplinary System or the Disciplinary System for Disruptive Conduct."""
    t = t.lower()
    t = t.replace(".", "")
    t = t.replace('"',"")
    t = t.replace('(',"")
    t = t.replace(')',"")
    t = t.replace(",", "")
    return t.split(" ")

def main():
    tuple_len = 1 # number of characters in each chunk, 1-10
    token_len = 1 # number of chunks per token, 1-5
    regularizer = 0.01 # prevents model overfitting. increasing regularizer smooths predictions, giving more variation in output, 0-0.1
    do_softmax = False # when True, makes most likely prediction even more likely
    num_generated_tokens = 50 # how much to generate
    separator = " " # space for word level, empty string for character level

    text = get_text() # modify this for text input
    encoder, X, Y = tokenize(text, tuple_length=tuple_len, token_length=token_len)
    model = RidgeRegression()
    model.fit(X, Y, regularizer)
    print(generate(model, encoder, encoder.get_seed(token_len), num_generated_tokens, do_softmax=do_softmax, separator=separator))

if __name__ == "__main__":
    main()