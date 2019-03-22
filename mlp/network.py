"""This module implements two MLP (Multi-layer Perceptron) and some supporting
functionality.

The main classes are:
- MLPRegressor: A MLP for regression
- MLPClassifier: A MLP for classification

The MLP classes follow a design that is similar to a mix of the Keras API and the scikit-learn estimator API.
"""
import json

import numpy as np
from sklearn import utils

import mlp.layers
import mlp.losses
from mlp.losses import RMSE, CategoricalCrossEntropy, BinaryCrossEntropy


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


class MLP:
    def __init__(self, layers=None, learning_rate=1.0, momentum=0.9, loss_func=None):
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
        self.layers = layers

        for i, layer in enumerate(layers):
            if i > 0:
                self.layers[i - 1].is_output = False
                self.layers[i - 1].next_layer = layer

                if layer.n_inputs is None:
                    # We need to infer the input shape from the previous layer.
                    layer.n_inputs = self.layers[i - 1].n_units
                else:
                    assert layer.n_inputs == self.layers[i - 1].n_units, \
                        "The number of inputs for layer %d does not match the number of units in the previous layer " \
                        "(%d != %d)." % (len(self.layers), layer.n_inputs, self.layers[i - 1].n_units)
            else:
                assert layer.n_inputs is not None, "The number of inputs for the first layer must be explicitly specified."

            layer.network = self
            layer.is_output = True
            layer.initialise_weights()

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

            epoch_loss = epoch_loss_history.mean()  # / self.layers[-1].shape[-1]
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

            if early_stopping_threshold > 0 and epoch_loss < early_stopping_threshold:
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
        raise NotImplementedError

    def score(self, X, y):
        """Calculate the score for given feature and target data sets.

        Arguments:
            X: The feature data set.
            y: The target data set.

        Returns: The score of the MLP for the given data sets.
        """
        raise NotImplementedError

    def __str__(self):
        class_name = self.__class__.__name__
        layers = '[%s]' % ', '.join([str(layer) for layer in self.layers])
        lr = str(self.learning_rate)
        m = str(self.momentum)
        lf = str(self.loss_func)

        return '%s(layers=%s, learning_rate=%s, momentum=%s, loss_func=%s())' % (class_name, layers, lr, m, lf)

    def json(self):
        """Create a JSON representation of a layer.

        Returns: a JSON-convertable dictionary containing the parameters that describe the layer instance.
        """
        return dict(
            clf_type=self.__class__.__name__,
            layers=[layer.json() for layer in self.layers],
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            loss_func=str(self.loss_func)
        )

    def save(self, filename):
        """Save a MLP to disk.

        Arguments:
            filename: The path + filename indicating where to save the MLP.
        """
        with open(filename, 'w') as file:
            json.dump(self.json(), file)

    @staticmethod
    def from_json(json_dict):
        """Create a MLP object from JSON.

        Arguments:
            json_dict: The JSON dictionary from which to create the MLP object.

        Returns: The instantiated MLP object.
        """
        module = __import__(MLP.__module__)
        class_ = getattr(module.network, json_dict['clf_type'])
        layers = []

        for layer_params in json_dict['layers']:
            layer_class = getattr(mlp.layers, layer_params['layer_type'])
            layers.append(layer_class.from_json(layer_params))

        return class_(layers=layers,
                      learning_rate=json_dict['learning_rate'],
                      momentum=json_dict['momentum'],
                      loss_func=getattr(mlp.losses, json_dict['loss_func'])())

    @staticmethod
    def load(filename):
        """Load a MLP from disk.

        Arguments:
            filename: The path + filename indicating where to load the MLP from.

        Returns: A new MLP object.
        """
        with open(filename, 'r') as file:
            json_dict = json.load(file)

        return MLPRegressor.from_json(json_dict)


class MLPRegressor(MLP):
    """A MLP for regression tasks."""

    def predict(self, X):
        """Predict the targets for a given feature data set.

        Arguments:
            X: The feature data set.

        Returns: The predicted targets for the given feature data.
        """
        return self._forward(X)

    def score(self, X, y):
        y_pred = self.predict(X)

        # Pearson R coefficient.
        return np.mean((y - y.mean()) * (y_pred - y_pred.mean())) / np.sqrt(y.var() * y_pred.var())


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
            loss_func = CategoricalCrossEntropy()

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
        if isinstance(self.loss_func, BinaryCrossEntropy):
            y_pred = self.predict_proba(X)

            return np.where(y_pred > 0.5, np.ones_like(y_pred), np.zeros_like(y_pred))
        else:
            return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)

        if isinstance(self.loss_func, BinaryCrossEntropy):
            return np.mean(y == y_pred)
        else:
            return np.mean(y.argmax(axis=1) == y_pred)
