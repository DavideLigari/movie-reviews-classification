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


def logreg_train(data, steps=10000, lr=0.01, batch_size=2000):
    """Train a logistic regression classifier."""

    X = data[:, :-1]
    Y = data[:, -1]
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accs = []
    losses = []
    for step in range(steps):
        index = np.random.choice(data.shape[0], batch_size, replace=False)
        newX = X[index]
        newY = Y[index]
        P = logreg_inference(newX, w, b)
        if step % 10000 == 0:
            loss = cross_entropy(P, newY)
            prediction = (P > 0.5)
            accuracy = (prediction == newY).mean()
            accs.append(accuracy)
            losses.append(loss)
            print(step)
        grad_w = ((P - newY) @ newX) / batch_size
        grad_b = (P - newY).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, accs, losses
