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

        self.prev_input = None
        self.W_0 = np.random.uniform(-0.3, 0.3, (n_hidden, n_inputs))
        self.b_0 = np.random.uniform(-0.3, 0.3, n_hidden)
        self.hidden_activation = None

        self.W_1 = np.random.uniform(-0.3, 0.3, (n_outputs, n_hidden))
        self.b_1 = np.random.uniform(-0.3, 0.3, n_outputs)
        self.output_activation = None

    @staticmethod
    def sigmoid(x):
        z = np.exp(x)

        return z / (z + 1)

    @staticmethod
    def dsigmoid(y):
        return y * (1 - y)

    def forward(self, x):
        self.prev_input = x

        output = self.W_0.dot(x) + self.b_0
        output = MLP.sigmoid(output)
        self.hidden_activation = output

        output = self.W_1.dot(output) + self.b_1
        output = MLP.sigmoid(output)
        self.output_activation = output

        return output

    def backward(self, y, y_pred):
        error = y - y_pred
        output_error = error * MLP.dsigmoid(self.output_activation)
        hidden_error = output_error.dot(self.W_1) * MLP.dsigmoid(self.hidden_activation)

        dW_1 = self.learning_rate * output_error * self.hidden_activation
        dW_0 = self.learning_rate * hidden_error * self.prev_input

        print(output_error, dW_1)
        print(hidden_error, dW_0)

        self.W_1 -= dW_1
        self.W_0 -= dW_0


if __name__ == '__main__':
    np.random.seed(42)

    X, y = get_XOR()
    mlp = MLP(X.shape[1], 2, 1, 0.1)

    n_epochs = 10

    for epoch in range(n_epochs):
        print('Epoch %d of %d' % (epoch + 1, n_epochs))

        for instance, label in zip(*get_XOR()):
            label_pred = mlp.forward(instance)
            rmse = np.sqrt(np.square(label - label_pred))
            print('Pattern: %s - Label: %d - Predicted Label: %.4f - RMSE: %.4f' % (instance, label, label_pred, rmse))
            mlp.backward(label, label_pred)
