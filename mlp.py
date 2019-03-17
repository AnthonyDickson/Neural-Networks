"""TODO: Add docstrings!"""

import numpy as np

from activation_functions import Identity


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
    else:  # mini-batch sgd
        n_batches, last_batch_size = divmod(N, batch_size)
        n_batches = n_batches + (1 if last_batch_size > 0 else 0)

    ix = 0

    for batch_i in range(n_batches):
        batch_start = ix
        batch_end = ix + batch_size

        if batch_end > N:
            batch_end = ix + last_batch_size

        yield batch_i, X[batch_start:batch_end], y[batch_start:batch_end]

        ix += batch_size


class Layer:
    def __init__(self, n_units, n_inputs=None, activation_func=None):
        self.n_inputs = n_inputs
        self.n_units = n_units

        self.W = None
        self.b = None
        self.prev_dW = None
        self.prev_db = None

        self.prev_input = None
        self.activation_value = None
        self.preactivation_value = None

        self.activation_func = activation_func if activation_func is not None else Identity()

        self.is_output = False
        self.network = None
        self.next_layer = None

    def initalise_weights(self):
        self.W = np.random.normal(0, 1, (self.n_inputs, self.n_units)) * np.sqrt(1.0 / self.n_inputs)
        self.b = np.random.normal(0, 1, (1, self.n_units))
        self.prev_dW = np.zeros_like(self.W)
        self.prev_db = np.zeros_like(self.b)

    @property
    def shape(self):
        return self.W.shape

    def forward(self, X):
        self.prev_input = X

        output = np.matmul(X, self.W) + self.b
        self.preactivation_value = output

        output = self.activation_func(output)
        self.activation_value = output

        return output

    def backward(self, errors):
        N = errors.shape[0]

        if self.is_output:
            delta = errors
        else:
            delta = errors.dot(self.next_layer.W.T)

        delta *= self.activation_func.derivative(self.activation_value)

        dW = self.network.learning_rate * np.matmul(self.prev_input.T, delta) + self.network.momentum * self.prev_dW
        db = self.network.learning_rate * delta + self.network.momentum * self.prev_db

        dW_mean = dW / N
        db_mean = db.mean(axis=0)

        self.W += dW_mean
        self.b += db_mean

        self.prev_dW = dW_mean
        self.prev_db = db_mean

        return delta


class MLP:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = []

    def add(self, layer):
        if len(self.layers) > 0:
            self.layers[-1].is_output = False
            self.layers[-1].next_layer = layer

            if layer.n_inputs is None:
                # We need to infer the input shape from the previous layer.
                layer.n_inputs = self.layers[-1].n_units
        else:
            assert layer.n_inputs is not None, "The number of inputs for the first layer must be explicitly specified."

        layer.network = self
        layer.is_output = True
        layer.initalise_weights()

        self.layers.append(layer)

    def _forward(self, X):
        assert len(self.layers) > 0, "The MLP needs at least one layer, however it currently has zero!"

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def _backward(self, errors):
        error_grad = errors

        for layer in reversed(self.layers):
            error_grad = layer.backward(error_grad)

    def fit(self, X, y, n_epochs=100, batch_size=-1, early_stopping_threshold=-1, early_stopping_min_improvement=1e-5):
        best_score = 2 ** 31 - 1
        epochs_no_improvement = 0
        error_history = []

        for epoch in range(n_epochs):
            epoch_error_history = np.array([])

            for _, X_batch, y_batch in _generate_minibatches(X, y, batch_size):
                target_pred = self._forward(X_batch)
                errors = y_batch - target_pred
                self._backward(errors)
                epoch_error_history = np.append(epoch_error_history, errors)

            rmse = np.sqrt(np.mean(np.square(epoch_error_history)))
            error_history.append(rmse)

            print('Epoch %d of %d - RMSE: %.4f' % (epoch + 1, n_epochs, rmse))

            if best_score - rmse > early_stopping_min_improvement:
                best_score = rmse
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if early_stopping_threshold > 0 and epochs_no_improvement > early_stopping_threshold:
                print('Stopping early.')
                break

        return error_history

    def predict(self, X):
        return self._forward(X)

    def score(self, X, y):
        y_pred = self.predict(X)

        return np.sqrt(np.mean(np.square(y - y_pred)))
