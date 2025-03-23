# Everything Generates Text

Because every ML model can be forced to generate text. And Jon Ma is the Best. And Ronik too, lol.

## Thanks for that introduction, Jon. I'll take it from here.

LLMs, such as ChatGPT, don't understand language. They aren't having real conversations with you. They can't construct meaning from words. They know nothing.

So, how do they do what they do? Simple. They are next-word* predictors. 

LLMs take whatever input text they are given, and they try to predict what the next word should be. Then, they add the predicted word to their input text and repeat the process. Inside the LLM, it looks something like this:

1. Once -> [upon, 57% probability]
2. Once upon -> [a, 72% probability]
3. Once upon a -> [time, 86% probability]

And so on. This is called autoregressive prediction, and it relies on one, massive assumption about language: all the words that come before give you enough information to guess what word comes next.

So far, this assumption has worked pretty well, but there's one problem. Because the software and hardware necessary to run LLMs is so incredibly complicated, companies and hucksters argue their machines are capable of actually thinking. Of reasoning. Of doing what human brains do. And this is simply false.

LLMs are no more capable of cognition than a shuffled deck of cards. Sure, sometimes you might see interesting patterns, but that's only because you gave the patterns meaning, not because the LLMs are doing anything special. LLMs generate sequences of words. You give those words meaning.

In order to demonstrate that text generation is not special, I wanted to create something that shows how unspectacular generating text really is. And so, I present, Everything Generates Text.

## Premise

LLMs are very fancy, expensive machine learning (ML) models, but ML models don't need to be fancy or expensive to generate text. In fact, I believe any ML model can be coerced into generating text, because it's not the machine that's special, it's just about training on the right data.

### Drawing Lines

Let's start with the most basic ML model: linear regression. You are absolutely familiar with linear regression. If you've ever seen a bunch of points scattered on a plot, and a trend line drawn through those points, congrats! You witnessed ML in action. Linear regression really is just drawing lines through data. So, if we can use this to generate text, then anything can generate text, right?

Well, it works. It's not fantastic, but sometimes it makes sense. To make this work, I essentially turned all the words into points scattered on a plot. I also needed to change our assumptions. Earlier, I said LLMs assume that if you know all the words before, then you can predict the next word. As it turns out, all the words requires a lot of computing power, so instead, I decided that if you know just one word that comes before, then you can probably guess the next word alright. I could have used all the words, but I wanted this to be fast, not useful. That, and my one-hot encoding scheme was far from optimal.

The same worked for logistic regression, if you care about those. It was essentially the same as the linear regression.

### Decisions Decisions

Decision trees are another ML model that are as simple as they sound. You have some decision you need to make, so you ask yes/no questions until you arrive at an answer. For example, you might have a decision tree for determining if you need an umbrella. It may ask, "is it raining?" and if you say yes, then it will say "yes, you need an umbrella." That's it. That's all a decision tree is. And somehow, it generates language.

Rather than predicting entire words, this time, I decided to predict one letter at a time, so the decision tree is just trying to decide which letter to write next. I give it a few of the letters that come before (not even a whole word) and it asks questions like "is the first letter an 'a'" and uses that to figure out which letter to write. The math for this was more intense than I expected, partly because I designed inefficient algorithms and my own version of formal logic to make the trees more interesting, but they work, not only writing actual words but coherent phrases.

You can also put together multiple trees and make a forest, just don't forget to see the forest for the trees.

### Howdy, Neighbor

In ML, cheating off your neighbor in class is encouraged. In fact, they designed a whole model off that premise. It's called k-Nearest Neighbors, or kNN. With this model, you use the training data directly, and every time you make a prediction, you just ask, "who am I closest to, and what answer did they get?" Then, you write their answer as your own. That's it. That's the whole model.

Here, I basically treated language like a multiple choice test where you need to guess the next letter. Using this, our model could cheat off its neighbors and, like all the other models, generate text.

### Chains

I've done Markov chain text generators to death, so if you care about this one, just check out one of my other projects on the subject. [fun website](https://ronikbhaskar.github.io/text_generator/) [brief summary](https://ronikbhaskar.github.io/projects/text_generator)

### Oh, You Poor, Naive Bayes

Naive Bayes is just a dumb version of a very powerful statistical model. It's naive because you assume there is no bias in what data you might ever see, ever, and that all the data is completely unrelated, somehow. You shouldn't assume these things, but I did, because it makes for a very simple ML model. Like the linear regression, it tries to predict the next word, but this time, it does so by checking each possible word and asking, "does this word usually come after the one I'm looking at?" and "is this word rare or common?"

By combining those two questions, it figures out which word is most likely. Surprisingly, this model tends to use uncommon words more often than the other models.

## The Proof of the Pudding is in the Eating

I wouldn't be mad if you if you didn't believe me. I mean, I just said that I made text generators from some very absurd ideas, like drawing lines and cheating on a test. So, to assuage any concerns you may have, I tested all of our models on their descriptions, and here is what they had to say:

Linear Regression:
> Be fast, right. I also needed to be fast, and a trend line drawn through those points, then you can use this to change our assumptions. I wanted this to generate text, not useful. It makes sense. To change our assumptions. You know just drawing lines through those points scattered on a bunch of points scattered on a plot. Earlier, I essentially the same as it turns out, I also needed to generate text, I essentially turned all the words before, all the most basic ml in

Logistic Regression:
> Let's start with the words requires a plot, I also needed to make this to make this to change our assumptions. I also needed to change our assumptions. I also needed to make this to generate text, then you can predict the words into points scattered on a plot, congrats. You know all the words requires a plot. I could have used all the words before, I decided that if you know just drawing lines through those points scattered on a plot. If you can generate text, I also needed to

Decision Tree:
> Decision you arrive at an answer. For example, you say yes, then it will say "yes, this time, I designed inefficient algorithms and make a forest for determining?" and my own version trees are as the decided to predicting actual words but coherent phrases. You can also put together ML model that to figure out which letter to writing actual word) and if you need an umbrella. It may ask, "is it raining?" and uses that come before (not even a whole words, this was more interesting, but coherent phrases.

k-Nearest Neighbors:
> In ML, cheating off your own. That's it. That's the whole model off its neighbors and, like a multiple choice test where you use their answer as your own. That's it. That's it. That's the next letter. Using this model, you need to guess the training data directly, and what answer did they designed a whole models, generate text.eir answer did they designed a whole models, generate text.letter. Using this, our model. Here, I basically treated language like a multiple choice test where you just ask, "wh

Naive Bayes:
> Often than the one I'm looking at. Like the one I'm looking at. Surprisingly, somehow. Like the next word usually come after the one I'm looking at. Like the linear regression, but I did, but I did, ever see, somehow. Like the data is completely unrelated, does so by checking each possible word rare or common. Like the next word usually come after the next word rare or common. It's naive because you might ever see, but I did, somehow. Surprisingly, somehow. Surprisingly

### Authors

Jon Ma and Ronik Bhaskar

### Notes

*technically, they predict the next token, not word, but for our purposes, they are functionally the same.