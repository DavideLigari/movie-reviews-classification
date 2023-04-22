import numpy as np


def train_nb(X, Y):
    """Train a binary NB classifier."""
    # + 1 for the Laplacian smoothing
    pos_p = X[Y == 1, :].sum(0) + 1
    pos_p = pos_p / pos_p.sum()
    neg_p = X[Y == 0, :].sum(0) + 1
    neg_p = neg_p / neg_p.sum()
    w = np.log(pos_p) - np.log(neg_p)
    # Estimate P(0) and P(1) and compute b
    pos_prior = Y.mean()
    neg_prior = 1 - pos_prior
    b = np.log(pos_prior) - np.log(neg_prior)
    return w, b


def inference_nb(X, w, b):
    """probability to be positive"""
    logits = X @ w + b
    return logits


def train_classifier(data):
    X = data[:, :-1]
    Y = data[:, -1]
    w, b = train_nb(X, Y)
    return w, b


def inference_classifier(data, w, b):
    X = data[:, :-1]
    Y = data[:, -1]
    logits = inference_nb(X, w, b)
    predictions = (logits > 0).astype(int)
    accuracy = (predictions == Y).mean()
    return logits, predictions, accuracy*100


# This part detects the most relevant words for the classifier.
def get_most_relevant_words(w, voc, data):
    f = open("vocabulary.txt")
    voc = f.read().split()
    f.close()
    X = data[:, :-1]
    Y = data[:, -1]
    indices = w.argsort()
    positive_words = []
    negative_words = []
#  NEGATIVE WORDS
    for i in indices[:20]:
        negative_words[voc[i], w[i]]

# POSITIVE WORDS
    for i in indices[-20:]:
        positive_words[voc[i], w[i]]
    return positive_words, negative_words
