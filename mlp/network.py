"""This module implements two MLP (Multi-layer Perceptron) and some supporting
functionality.

The main classes are:
- MLPRegressor: A MLP for regression
- MLPClassifier: A MLP for classification
- DenseLayer: A fully connected neural network layer.
"""

import numpy as np
from sklearn import utils

from mlp.activation_functions import Identity
from mlp.losses import RMSE, CrossEntropy


def _generate_minibatches(X, y, batch_size=32, shuffle=False):
    """Generate mini-batches from the given X-y sets.



    Arguments:
        X: The feature data set.
        y: The target data set.
        batch_size: The size of the batches to generate from X and y.
                    If this parameter is set to negative one, then the batch
                    size is set the length of the entire X set - which is
                    equivalent to batch SGD.
                    If this parameter is set to one, then using the batches
                    generated for training is equivalent to performing standard
                    SGD.

        shuffle: Whether or not to shuffle the data.
    """
    if shuffle:
        X, y = utils.shuffle(X, y)

    N = len(X)
    assert N == len(y)
    assert batch_size == -1 or batch_size > 0, "Invalid batch size."

    batch_size = min(batch_size, N)

    if batch_size == -1:  # batch sgd
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


class DenseLayer:
    """A fully connected layer in a neural network."""

    def __init__(self, n_units, n_inputs=None, activation_func=None):
        """Create a fully connected layer.

        Arguments;
            n_units: How many units (or artificial neurons) the layer should have.

            n_inputs: How many inputs the layer should accept. If this is set to
            None then the input size will be inferred from the previous layer.

            activation_func: The activation function to use for this layer.
            If set the None this defaults to the identity function.
        """
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

    def initialise_weights(self):
        """Create and initialise the weight and bias matrices."""
        self.W = np.random.normal(0, 1, (self.n_inputs, self.n_units)) * np.sqrt(1.0 / self.n_inputs)
        self.b = np.random.normal(0, 1, (1, self.n_units))
        self.prev_dW = np.zeros_like(self.W)
        self.prev_db = np.zeros_like(self.b)

    @property
    def shape(self):
        """Get the shape of the layer,

        Returns: A 2-tuple containing the number of inputs for each unit and
        the number of units in the layer.
        """
        return self.W.shape

    def forward(self, X):
        """Perform a forward pass of the MLP.

        Returns: The activation of the layer.
        """
        self.prev_input = X

        output = np.matmul(X, self.W) + self.b
        self.preactivation_value = output

        output = self.activation_func(output)
        self.activation_value = output

        return output

    def backward(self, error_term):
        """Perform a backward pass of the layer (i.e. back propagate error) and
        update weights and biases.

        Arguments:
            error_term: The error term for the output layer, or the deltas of
            the previous layer in the case of a hidden layer.

        Returns the calculated deltas for the given layer.
        """
        N = error_term.shape[0]

        if self.is_output:
            delta = error_term
        else:
            delta = error_term.dot(self.next_layer.W.T)

        delta *= self.activation_func.derivative(self.activation_value)

        dW = self.network.learning_rate * np.matmul(self.prev_input.T, delta) \
             + self.network.momentum * self.prev_dW
        db = self.network.learning_rate * delta + \
             self.network.momentum * self.prev_db

        dW_mean = dW / N
        db_mean = db.mean(axis=0)

        self.W += dW_mean
        self.b += db_mean

        self.prev_dW = dW_mean
        self.prev_db = db_mean

        return delta


class MLPRegressor:
    """A MLP for regression tasks."""

    def __init__(self, layers, learning_rate=1.0, momentum=0.9, loss_func=None):
        """Create a MLP.

        Arguments:
            layers: A list of the layers to use for the network. See the
            documentation on DenseLayer for more details.

            learning_rate: The learning constant which controls how big of a
            step is taken during SGD.

            momentum: The momentum constant which controls how much of the
            previous weight changes are carried over to the next update.

            loss_func: The loss function to be used for evaluating the
            performance of the MLP.
        """
        assert len(layers) > 0, "You need to define at least one layer for the network."

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_func = loss_func if loss_func is not None else RMSE()
        self.layers = []

        for layer in layers:
            self._add(layer)

    def _add(self, layer):
        """Add a layer to the MLP.

        The layer added last is set as the output layer.

        Arguments:
            layer: The layer to add to the MLP.
        """
        if len(self.layers) > 0:
            self.layers[-1].is_output = False
            self.layers[-1].next_layer = layer

            if layer.n_inputs is None:
                # We need to infer the input shape from the previous layer.
                layer.n_inputs = self.layers[-1].n_units
            else:
                assert layer.n_inputs == self.layers[-1].n_units, \
                    "The number of inputs for layer %d does not match the number of units in the previous layer " \
                    "(%d != %d)." % (len(self.layers), layer.n_inputs, self.layers[-1].n_units)
        else:
            assert layer.n_inputs is not None, "The number of inputs for the first layer must be explicitly specified."

        layer.network = self
        layer.is_output = True
        layer.initialise_weights()

        self.layers.append(layer)

    def _forward(self, X):
        """Perform a forward pass of the MLP.

        Arguments:
            X: The feature data set.

        Returns; The output of the MLP.
        """
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def _backward(self):
        """Perform a backward pass of the MLP (i.e. back propagate error) and
        update weights and biases.
        """
        error_grad = self.loss_func.grad

        for layer in reversed(self.layers):
            error_grad = layer.backward(error_grad)

    def fit(self, X, y, n_epochs=100, batch_size=-1, early_stopping_patience=-1,
            early_stopping_min_improvement=1e-5, early_stopping_threshold=1e-2,
            log_verbosity=1):
        """Fit/train the MLP on the given data sets.

        Arguments:
            X: The feature data set.
            y: The target data set.
            n_epochs: How many epochs to train the MLP for.
            batch_size: The size of the batches to use for training.
            If this is set to -1, batch SGD is performed; if this is set to 1,
            standard SGD is performed; otherwise mini-batch SGD is performed.

            early_stopping_patience: The number of epochs after no improvement
            in loss is observed which training should be stopped early. If set
            to any number less than one, early stopping based on the change in
            loss is disabled.

            early_stopping_min_improvement: The minimum change of loss that is
            to be considered an improvement.

            early_stopping_threshold: The learning criterion, or the target
            loss. Training is stopped once the loss is less than the criterion.

            log_verbosity: How often to log training progress. Large values
            will make training progress be logged less frequently.
        """
        min_loss = 2 ** 31 - 1
        epochs_no_improvement = 0
        loss_history = []

        for epoch in range(n_epochs):
            epoch_loss_history = np.array([])

            for _, X_batch, y_batch in _generate_minibatches(X, y, batch_size):
                target_pred = self._forward(X_batch)
                loss = self.loss_func(y_batch, target_pred)
                epoch_loss_history = np.append(epoch_loss_history, loss)

                self._backward()

            epoch_loss = epoch_loss_history.mean()
            loss_history.append(epoch_loss)

            if log_verbosity > 0 and epoch % log_verbosity == 0:
                print('Epoch %d of %d - Loss: %.4f' % (epoch + 1, n_epochs, loss_history[-1]))

            if min_loss - epoch_loss > early_stopping_min_improvement:
                min_loss = epoch_loss
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if early_stopping_patience > 0 and epochs_no_improvement > early_stopping_patience:
                if log_verbosity > 1 and epoch % log_verbosity != 0:
                    print('Epoch %d of %d - Loss: %.4f' % (epoch + 1, n_epochs, loss_history[-1]))

                print('Stopping early - loss has stopped improving.')
                break

            if epoch_loss < early_stopping_threshold:
                if log_verbosity > 1 and epoch % log_verbosity != 0:
                    print('Epoch %d of %d - Loss: %.4f' % (epoch + 1, n_epochs, loss_history[-1]))

                print('Stopping early - reached target error criterion.')
                break

        return loss_history

    def predict(self, X):
        """Predict the targets for a given feature data set.

        Arguments:
            X: The feature data set.

        Returns: The predicted targets for the given feature data.
        """
        return self._forward(X)

    def score(self, X, y):
        """Calcaulate the loss for given feature and target data sets.

        Arguments:
            X: The feature data set.
            y: The target data set.

        Returns: The loss of the MLP for the given data sets.
        """
        y_pred = self.predict(X)

        return self.loss_func(y, y_pred)


class MLPClassifier(MLPRegressor):
    """A MLP for classification tasks."""

    def __init__(self, layers, learning_rate=1.0, momentum=0.9, loss_func=None):
        """Create a MLP.

        Arguments:
            layers: A list of the layers to use for the network. See the
            documentation on DenseLayer for more details.

            learning_rate: The learning constant which controls how big of a
            step is taken during SGD.

            momentum: The momentum constant which controls how much of the
            previous weight changes are carried over to the next update.

            loss_func: The loss function to be used for evaluating the
            performance of the MLP. This defaults to cross entropy loss.
        """
        if loss_func is None:
            loss_func = CrossEntropy()

        super().__init__(layers, learning_rate, momentum, loss_func)

    def predict_proba(self, X):
        """Predict the targets for a given feature data set.

        Arguments:
            X: The feature data set.

        Returns: The predicted targets for the given feature data as a
        probability distribution.
        """
        return self._forward(X)

    def predict(self, X):
        """Predict the targets for a given feature data set.

        Arguments:
            X: The feature data set.

        Returns: The predicted targets for the given feature data.
        """
        return np.argmax(self.predict_proba(X), axis=0)
