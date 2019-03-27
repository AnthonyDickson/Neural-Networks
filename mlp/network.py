"""This module implements two MLP (Multi-layer Perceptron) and some supporting
functionality.

The main classes are:
- MLPRegressor: A MLP for regression
- MLPClassifier: A MLP for classification

The MLP classes follow a design that is similar to a mix of the Keras API and the scikit-learn estimator API.
"""
import gzip
import json
import pickle

import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split

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


class EarlyStopping:
    """An object that implements early stopping.

    Training can be stopped early based on the lack of improvement of loss or upon reaching a target loss,
    """

    def __init__(self, patience=10, min_improvement=1e-5, criterion=0.999):
        """
        Create an EarlyStopping object.

        Arguments:
            patience: The number of epochs after no improvement in loss is observed which training should be stopped
            early. If set to any number less than one, early stopping based on the change in loss is disabled.

            min_improvement: The minimum change of loss that is to be considered an improvement.

            criterion: The learning criterion, or the target score. Training is stopped once the score is greater than the
            criterion.
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.criterion = criterion
        self.epochs_no_improvement = 0
        self.min_loss = 2 ** 32 - 1
        self.reason = ''
        self.last_loss = 2 ** 32 - 1
        self.last_score = -2 ** 32 + 1

    @property
    def should_stop(self):
        """Check whether training should stop.

        Returns: True if training should stop, False otherwise.
        """
        if self.patience > 0 and self.epochs_no_improvement > self.patience:
            self.reason = 'loss has stopped improving'

            return True
        elif self.criterion > 0 and self.last_score > self.criterion:
            self.reason = 'reached target score criterion'

            return True
        else:
            return False

    def update(self, loss, score):
        """Update the state of the early stopping object.

        Arguments:
            loss: The loss to measure.
            score: The score to measure.
        """
        self.last_score = score

        if self.min_loss - loss > self.min_improvement:
            self.min_loss = loss
            self.epochs_no_improvement = 0
        else:
            self.epochs_no_improvement += 1


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

    def _forward(self, X, is_training=True):
        """Perform a forward pass of the MLP.

        Arguments:
            X: The feature data set.
            is_training: Whether or not the forward pass is being done during training or not.

        Returns; The output of the MLP.
        """
        output = X

        for layer in self.layers:
            output = layer.forward(output, is_training)

        return output

    def _backward(self):
        """Perform a backward pass of the MLP (i.e. back propagate error) and
        update weights and biases.
        """
        error_grad = self.loss_func.grad

        for layer in reversed(self.layers):
            error_grad = layer.backward(error_grad)

    def fit(self, X, y, val_set=0.0, n_epochs=100, batch_size=-1, shuffle_batches=True,
            early_stopping=None, log_verbosity=1):
        """Fit/train the MLP on the given data sets.

        Arguments:
            X: The feature data set for training the MLP on.
            y: The target data set for training the MLP on.
            val_set: The data set to be used for validation. This can be an integer indicating how many samples from the
            training sets to use for validation; or it can be a ratio indicating what proportion of the training data to
            use for validation; or it can be a tuple containing the X and y validation sets.

            n_epochs: How many epochs to train the MLP for.

            batch_size: The size of the batches to use for training.
            If this is set to -1, batch SGD is performed; if this is set to 1,
            standard SGD is performed; otherwise mini-batch SGD is performed.

            early_stopping: See `EarlyStopping`. If set to None then early stopping is not used.

            log_verbosity: How often to log training progress. Large values
            will make training progress be logged less frequently.

            shuffle_batches: Whether or not to shuffle the batches each epoch.

        Returns: A 4-tuple of training loss, training score, validation loss, and validation score history data.
        """
        assert len(y.shape) == 2, 'The target vector `y` must either be a column vector or matrix, ' \
                                  'got a row vector instead.\n' \
                                  'If `y` has one \'feature\' then you can reshape y with `y.reshape(-1, 1)`, ' \
                                  'otherwise if `y` is a single sample then can reshape y with `y.reshape(1, -1)`.'

        train_loss_history = []
        train_score_history = []
        val_loss_history = []
        val_score_history = []

        X_train, X_val, y_train, y_val = self._train_val_split(X, y, val_set)

        for epoch in range(n_epochs):
            epoch_train_loss_history = np.array([])
            epoch_train_score_history = np.array([])

            for _, X_batch, y_batch in _generate_minibatches(X_train, y_train, batch_size, shuffle=shuffle_batches):
                target_pred = self._forward(X_batch)
                loss = self.loss_func(y_batch, target_pred)
                epoch_train_loss_history = np.append(epoch_train_loss_history, loss)
                score = self.score(X_batch, y_batch)
                epoch_train_score_history = np.append(epoch_train_score_history, score)

                self._backward()

            train_loss_history.append(epoch_train_loss_history.mean())
            train_score_history.append(epoch_train_score_history.mean())

            if val_set:
                val_loss_history.append(self.loss_func(y_val, self._forward(X_val, is_training=False)).mean())
                val_score_history.append(self.score(X_val, y_val))
            else:
                val_loss_history.append(np.nan)
                val_score_history.append(np.nan)

            if log_verbosity > 0 and epoch % log_verbosity == 0:
                print('epoch %d of %d - loss: %.4f - score: %.4f - val_loss: %.4f - val_score: %.4f'
                      % (epoch + 1, n_epochs, train_loss_history[-1], train_score_history[-1],
                         val_loss_history[-1], val_score_history[-1]))

            if early_stopping:
                if val_set:
                    early_stopping.update(val_loss_history[-1], val_score_history[-1])
                else:
                    early_stopping.update(train_loss_history[-1], train_score_history[-1])

                if early_stopping.should_stop:
                    if log_verbosity > 0:
                        print('epoch %d of %d - loss: %.4f - score: %.4f - val_loss: %.4f - val_score: %.4f'
                              % (epoch + 1, n_epochs, train_loss_history[-1], train_score_history[-1],
                                 val_loss_history[-1], val_score_history[-1]))
                        print('Stopping early - %s.' % early_stopping.reason)

                    break

        return train_loss_history, train_score_history, val_loss_history, val_score_history

    def _train_val_split(self, X, y, val_set):
        if type(val_set) is int or type(val_set) is float:
            if (type(val_set) is int and val_set < len(np.unique(y))) or val_set * len(X) < len(np.unique(y)):
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_set)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_set, stratify=y)
        elif type(val_set) is tuple:
            X_train, y_train = X, y
            X_val, y_val = val_set
        else:
            raise ValueError('Invalid type `%s` for val_set, expected int, float or tuple.' % type(val_set))

        return X_train, X_val, y_train, y_val

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

    def to_json(self):
        """Create a JSON representation of a layer.

        Returns: a JSON-convertible dictionary containing the hyper-parameters that describe the MLP.
        """
        return dict(
            clf_type=self.__class__.__name__,
            layers=[layer.to_json() for layer in self.layers],
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            loss_func=str(self.loss_func)
        )

    def save(self, filename):
        """Save a MLP to disk.

        Weights are not save. For restoring weights see `save_weights`.

        Arguments:
            filename: The path + filename indicating where to save the MLP.
        """
        with open(filename, 'w') as file:
            json.dump(self.to_json(), file)

    def save_weights(self, filename):
        """Save the weights and bias of a MLP to disk.

        Arguments:
            filename: The path + filename indicating where to save the MLP parameters.
        """
        with gzip.open(filename, 'w') as file:
            pickle.dump([(layer.W, layer.b) for layer in self.layers], file)

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

        Weights are not loaded. For restoring weights see `load_weights`.

        Arguments:
            filename: The path + filename indicating where to load the MLP from.

        Returns: A new MLP object.
        """
        with open(filename, 'r') as file:
            json_dict = json.load(file)

        return MLPRegressor.from_json(json_dict)

    def load_weights(self, filename):
        """Load the weights and bias of a MLP from disk.

        Arguments:
            filename: The path + filename indicating where to load the MLP parameters from.
        """
        with gzip.open(filename, 'r') as file:
            weights_bias = pickle.load(file)

        assert len(weights_bias) == len(self.layers), \
            "Layer count mismatch. This MLP has %d layers, however the file '%s' indicates %d layers." \
            % (len(self.layers), filename, len(weights_bias))

        for (weights, bias), layer in zip(weights_bias, self.layers):
            layer.W = weights
            layer.b = bias


class MLPRegressor(MLP):
    """A MLP for regression tasks."""

    def predict(self, X):
        """Predict the targets for a given feature data set.

        Arguments:
            X: The feature data set.

        Returns: The predicted targets for the given feature data.
        """
        return self._forward(X, is_training=False)

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
        return self._forward(X, is_training=False)

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
