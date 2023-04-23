import numpy as np


def logreg_inference(X, w, b):
    """Return the probability of being positive."""
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


def cross_entropy(P, Y):
    """Return the cross-entropy loss."""
    P = np.clip(P, 0.0001, 0.9999)
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


def logreg_train(data, steps=10000, lr=0.01):
    """Train a logistic regression classifier."""
    X = data[:, :-1]
    Y = data[:, -1]
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accs = []
    losses = []
    for step in range(steps):
        P = logreg_inference(X, w, b)
        if step % 1000 == 0:
            loss = cross_entropy(P, Y)
            prediction = (P > 0.5)
            accuracy = (prediction == Y).mean()
            accs.append(accuracy)
            losses.append(loss)
        grad_w = ((P - Y) @ X) / m
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, accs, losses
