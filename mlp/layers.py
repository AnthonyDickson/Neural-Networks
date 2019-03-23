"""This module implements layers that are used in artificial neural networks.

All layers provide the methods `forward(X)` and `backward` to encapsulate to forward and back propagation processes.
"""

import numpy as np

import mlp.activation_functions
from mlp.activation_functions import Identity


class Layer:
    """An abstraction of a layer in a neural network."""

    def __init__(self, n_units, n_inputs=None):
        """Create"""
        self.n_inputs = n_inputs
        self.n_units = n_units

    def forward(self, X):
        """Perform a forward pass of a layer.

        Arguments:
            X: The input to the layer.

        Returns: The result of the forward pass of the layer.
        """
        raise NotImplementedError

    def backward(self, error_term):
        """Perform a backward pass of a layer (i.e. back propagate error) and
        update the layer's parameters.

        Arguments:
            error_term: The error term for the output layer, or the deltas of
            the previous layer in the case of a hidden layer.

        Returns the calculated deltas for the given layer.
        """
        raise NotImplementedError

    def __str__(self):
        return '%s(n_inputs=%d, n_units=%d)' % (self.__class__.__name__, self.n_inputs, self.n_units)

    def json(self):
        """Create a JSON representation of a layer.

        Returns: a JSON-convertable dictionary containing the parameters that describe the layer instance.
        """
        return dict(
            layer_type=self.__class__.__name__,
            n_inputs=self.n_inputs,
            n_units=self.n_units
        )

    @staticmethod
    def from_json(json_dict):
        """Create a layer object from JSON.

        Arguments:
            json_dict: The JSON dictionary from which to create the layer object.

        Returns: The instantiated layer object.
        """
        module = __import__(Layer.__module__)
        class_ = getattr(module.layers, json_dict['layer_type'])

        return class_(n_units=json_dict['n_units'],
                      n_inputs=json_dict['n_inputs'])


class DenseLayer(Layer):
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
        super().__init__(n_units, n_inputs)

        self.W = None
        self.b = None
        self.prev_dW = None
        self.prev_db = None

        self.activation_func = activation_func if activation_func is not None else Identity()

        self.prev_input = None
        self.activation_value = None
        self.preactivation_value = None

        self.is_output = False
        self.network = None
        self.next_layer = None

        self.initialised = False

    def initialise_weights(self):
        """Create and initialise the weight and bias matrices."""
        self.W = np.random.normal(0, 1, (self.n_inputs, self.n_units)) * np.sqrt(1.0 / self.n_inputs)
        self.b = np.random.normal(0, 1, (1, self.n_units))
        self.prev_dW = np.zeros_like(self.W)
        self.prev_db = np.zeros_like(self.b)
        self.initialised = True

    @property
    def shape(self):
        """Get the shape of the layer,

        Returns: A 2-tuple containing the number of inputs for each unit and
        the number of units in the layer.
        """
        return self.W.shape

    def forward(self, X):
        """Perform a forward pass of a layer.

        Arguments:
            X: The input to the layer.

        Returns: The activation of the layer.
        """
        self.prev_input = X

        output = np.matmul(X, self.W) + self.b
        self.preactivation_value = output

        output = self.activation_func(output)
        self.activation_value = output

        return output

    def backward(self, error_term):
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

        self.W -= dW_mean
        self.b -= db_mean

        self.prev_dW = dW_mean
        self.prev_db = db_mean

        return delta

    def __str__(self):
        return '%s(n_inputs=%d, n_units=%d, activation_func=%s())' \
               % (self.__class__.__name__, self.n_inputs, self.n_units, str(self.activation_func))

    def json(self):
        json_dict = super().json()
        json_dict['activation_func'] = str(self.activation_func)

        return json_dict

    @staticmethod
    def from_json(json_dict):
        return DenseLayer(n_units=json_dict['n_units'],
                          n_inputs=json_dict['n_inputs'],
                          activation_func=getattr(mlp.activation_functions, json_dict['activation_func'])())


# TODO: Implement gaussian noise layer.
class GaussianNoise(Layer):
    def __init__(self):
        raise NotImplementedError