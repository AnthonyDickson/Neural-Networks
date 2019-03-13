import matplotlib.pyplot as plt
import numpy as np


def get_XOR():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = [0, 1, 1, 0]

    return X, y


def plot_XOR():
    X, y = get_XOR()
    labels = np.unique(y)

    for label in labels:
        ix = np.where(y == label)
        plt.scatter(X[ix, 0], X[ix, 1], label=str(label))

    plt.title('Plot of XOR outputs')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend(title='XOR Output')
    plt.show()


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.W_0 = np.random.uniform(-0.3, 0.3, (n_inputs, n_hidden))
        self.b_0 = np.random.uniform(-0.3, 0.3, n_hidden)
        self.W_1 = np.random.uniform(-0.3, 0.3, (n_hidden, n_outputs))
        self.b_1 = np.random.uniform(-0.3, 0.3, n_outputs)

    @staticmethod
    def sigmoid(x):
        z = np.exp(x)

        return z / (z + 1)

    def forward(self, x):
        output = self.W_0.T.dot(x) + self.b_0
        output = MLP.sigmoid(output)
        output = self.W_1.T.dot(output) + self.b_1

        return output


if __name__ == '__main__':
    np.random.seed(42)

    X, y = get_XOR()
    mlp = MLP(X.shape[1], 2, 1, 0.1)

    for instance, label in zip(*get_XOR()):
        label_pred = mlp.forward(instance)
        rmse = np.sqrt(np.square(label - label_pred))
        print('Pattern: %s - Label: %d - Predicted Label: %.4f - RMSE: %.4f' % (instance, label, label_pred, rmse))
