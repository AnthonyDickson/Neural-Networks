import numpy as np


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.prev_input = None
        self.W_0 = np.random.normal(0, 1.0, (n_hidden, n_inputs)) * np.sqrt(1.0 / n_inputs)
        self.b_0 = np.random.normal(0, 1.0, n_hidden)
        self.hidden_activation = None

        self.W_1 = np.random.normal(0, 1.0, (n_outputs, n_hidden)) * np.sqrt(1.0 / n_hidden)
        self.b_1 = np.random.normal(0, 1.0, n_outputs)
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
        db_1 = self.learning_rate * output_error
        dW_0 = self.learning_rate * hidden_error * self.prev_input
        db_0 = self.learning_rate * hidden_error

        self.W_1 += dW_1
        self.b_1 += db_1
        self.W_0 += dW_0
        self.b_0 += db_0

    def fit(self, X, y, n_epochs):
        for epoch in range(n_epochs):
            print('Epoch %d of %d' % (epoch + 1, n_epochs))
            errors = []

            for instance, target in zip(X, y):
                target_pred = self.forward(instance)
                error = target - target_pred
                errors.append(error)
                print('Pattern: %s - Target: %d - Predicted Target: %.4f' % (
                    instance, target, target_pred))
                self.backward(target, target_pred)

            rmse = np.sqrt(np.mean(np.square(errors)))

            print('RMSE: %.4f' % rmse)
