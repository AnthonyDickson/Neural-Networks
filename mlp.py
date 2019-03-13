import numpy as np


def _generate_minibatches(X, y, batch_size=32):
    N = len(X)
    assert N == len(y)

    if batch_size == 1:  # standard sgd
        batch_size = 1
        n_batches = N
        last_batch_size = batch_size
    elif batch_size == -1:  # batch sgd
        batch_size = N
        n_batches = 1
        last_batch_size = batch_size
    else:  # minibatch sgd
        n_batches, last_batch_size = divmod(N, batch_size)
        n_batches = n_batches + (1 if last_batch_size > 0 else 0)

    ix = 0

    for batch_i in range(n_batches):
        batch_start = ix
        batch_end = ix + batch_size

        if batch_end > N:
            batch_end = ix + last_batch_size

        yield batch_i, X[batch_start:batch_end], y[batch_start:batch_end]


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
    def sigmoid(X):
        Z = np.exp(X)

        return Z / (Z + 1)

    @staticmethod
    def dsigmoid(Y):
        return Y * (1 - Y)

    def forward(self, X):
        self.prev_input = X

        output = X.dot(self.W_0) + self.b_0
        output = MLP.sigmoid(output)
        self.hidden_activation = output

        output = output.dot(self.W_1.T) + self.b_1
        output = MLP.sigmoid(output)
        self.output_activation = output

        return output

    def backward(self, errors):
        output_error = errors * MLP.dsigmoid(self.output_activation)
        hidden_error = output_error.dot(self.W_1) * MLP.dsigmoid(self.hidden_activation)

        dW_1 = self.learning_rate * output_error * self.hidden_activation
        db_1 = self.learning_rate * output_error
        dW_0 = self.learning_rate * hidden_error * self.prev_input
        db_0 = self.learning_rate * hidden_error

        self.W_1 += dW_1.mean(axis=0)
        self.b_1 += db_1.mean(axis=0)
        self.W_0 += dW_0.mean(axis=0)
        self.b_0 += db_0.mean(axis=0)

    def fit(self, X, y, n_epochs=100, batch_size=-1):
        for epoch in range(n_epochs):
            epoch_error_history = np.array([])

            for _, X_batch, y_batch in _generate_minibatches(X, y, batch_size):
                target_pred = self.forward(X_batch)
                errors = y_batch - target_pred
                self.backward(errors)
                epoch_error_history = np.append(epoch_error_history, errors)

            rmse = np.sqrt(np.mean(np.square(epoch_error_history)))

            print('Epoch %d of %d - RMSE: %.4f' % (epoch + 1, n_epochs, rmse))

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)

    def score(self, X, y):
        y_pred = self.predict_proba(X)

        return np.sqrt(np.mean(np.square(y - y_pred)))
