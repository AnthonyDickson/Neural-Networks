import matplotlib.pyplot as plt
import numpy as np


def load_iris():
    X = np.genfromtxt('data/iris/in.txt')
    y = np.genfromtxt('data/iris/teach.txt')

    return X, y

def load_XOR():
    X = np.genfromtxt('data/xor/in.txt')
    y = np.genfromtxt('data/xor/teach.txt')
    y = y.reshape(-1, 1)

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
