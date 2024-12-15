

from egt.utils.probability import make_distribution

import numpy as np

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # we ignore the marginal because it's a constant scalar
        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._means[idx, :] = X_c.mean(axis=0)
            self._vars[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples

    def predict(self, x):
        log_probs = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            # technically, this should be minus P(X), but that's a constant
            log_probs.append(posterior + prior)

        # for a strict Naive Bayes, you would just take the argmax of log probs
        log_probs = np.array(log_probs)
        log_probs += np.min(log_probs)
        temp = 1
        probs = np.exp(log_probs) / temp
        probs /= np.sum(probs)

        return np.random.choice(self._classes, size=1, p=probs)[0]
    
    def _pdf(self, class_idx, x):
        mean = self._means[class_idx]
        var = self._vars[class_idx]
        # neither of the +1's are actually in the formula for a gaussian distribution
        # the problem is the data is so sparse that sometimes a class only shows up once
        alpha = 0.05
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + alpha))
        denominator = np.sqrt(1 * np.pi * var + alpha)
        
        return numerator / denominator

        