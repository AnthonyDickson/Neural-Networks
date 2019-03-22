"""This module implements a few activation functions that are used in neural
networks.

The functions are called using the `__call__(X)` method, and the derivative of an activation function is given by the
`derivative(Y)` method.
"""

import numpy as np


class Activation:
    """An abstraction of activation functions.

    Activation functions provide two sets of functionality:
    1. Calculation of the output of the activation function.
    2. Calculation of the activation function's derivative,
    """

    def __call__(self, X):
        """Calculate the output of the activation function.

        Arguments;
            X: The input to the activation function.

        Returns: The activation value, i.e. the input transformed by the activation function.
        """
        raise NotImplementedError

    def derivative(self, Y):
        """Calculate the derivative of the activation function.

        Arguments;
            Y: The input that the derivative will be calculated with respect to.
            This is typically the previous output of the activation function.

        Returns: The derivative of the activation function w.r.t. the given input.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class Identity(Activation):
    """The identity function."""

    def __call__(self, X):
        return X

    def derivative(self, Y):
        return np.ones_like(Y)


class ReLU(Activation):
    """The rectified linear unit (ReLU) activation function."""

    def __call__(self, X):
        return np.where(X > 0, X, np.zeros_like(X))

    def derivative(self, Y):
        return np.where(Y > 0, np.ones_like(Y), np.zeros_like(Y))


class Sigmoid(Activation):
    """The sigmoid (or logistic) activation function."""

    def __call__(self, X):
        Z = np.exp(X)

        return Z / (Z + 1)

    def derivative(self, Y):
        return Y * (1 - Y)


class Softmax(Activation):
    """The softmax activation function."""

    def __call__(self, X):
        Z = np.exp(X)

        return Z / Z.sum(axis=1, keepdims=True)

    def derivative(self, Y):
        # return np.diagflat(Y) * np.dot(Y, Y.T)
        return np.ones_like(Y)  # TODO: Is this fine?
