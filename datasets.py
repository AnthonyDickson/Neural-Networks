import matplotlib.pyplot as plt
import numpy as np


def load_XOR():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = [[0],
         [1],
         [1],
         [0]]

    return X, y


def plot_XOR():
    X, y = load_XOR()
    labels = np.unique(y)

    for label in labels:
        ix = np.where(y == label)
        plt.scatter(X[ix, 0], X[ix, 1], label=str(label))

    plt.title('Plot of XOR outputs')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend(title='XOR Output')
    plt.show()