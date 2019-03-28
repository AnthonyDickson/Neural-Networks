import matplotlib.pyplot as plt
import numpy as np


def load_iris(data_dir='data/'):
    return load_dataset('iris', data_dir)


def load_XOR(data_dir='data/'):
    return load_dataset('xor', data_dir)


def load_dataset(dataset, data_dir='data/'):
    if not data_dir.endswith('/'):
        data_dir += '/'

    X = np.genfromtxt(data_dir + dataset + '/in.txt')
    y = np.genfromtxt(data_dir + dataset + '/teach.txt')

    if len(y.shape) == 1:
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
