"""Multinomial Naive Bayes from counts; no estimator or vectorizer library."""
from collections import Counter

import numpy as np


class NaiveBayes:
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, documents, labels):
        if not documents or len(documents) != len(labels):
            raise ValueError("Provide nonempty, aligned documents and labels")
        self.classes, class_counts = np.unique(labels, return_counts=True)
        self.words = sorted({word for doc in documents for word in doc})
        if not self.words:
            raise ValueError("Training vocabulary is empty")
        self.vocabulary = {word: i for i, word in enumerate(self.words)}
        class_index = {label: i for i, label in enumerate(self.classes)}
        self.counts = np.zeros((len(self.classes), len(self.words)), dtype=np.int64)
        for doc, label in zip(documents, labels):
            for word, count in Counter(doc).items():
                self.counts[class_index[label], self.vocabulary[word]] += count
        # P(c) = number of documents in c / all training documents.
        self.log_prior = np.log(class_counts / class_counts.sum())
        # P(w|c) = (count(w,c) + alpha) / (all tokens in c + alpha * |V|).
        smoothed = self.counts + self.alpha
        self.log_likelihood = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def scores(self, documents):
        scores = np.tile(self.log_prior, (len(documents), 1))
        for row, doc in enumerate(documents):
            for word, count in Counter(doc).items():
                column = self.vocabulary.get(word)
                if column is not None:  # Unseen words contribute nothing.
                    scores[row] += count * self.log_likelihood[:, column]
        return scores

    def predict(self, documents):
        return self.classes[self.scores(documents).argmax(axis=1)]
