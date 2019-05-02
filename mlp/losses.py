"""This module implements a few loss functions that are used in neural
networks.
"""

import numpy as np


class Loss:
    """An abstraction of loss functions.

    A loss function provides two sets of functionality:
    1. Calculation of loss given a set of ground truths and predictions
    2. Calculation of the gradient of the previously calculated loss.
    """
    EPSILON = 1e-15

    def __init__(self):
        self.grad = None  # the gradient, or derivative of the loss function w.r.t the output.

    def __call__(self, y, y_pred):
        """Calculate loss.

        Arguments:
            y: The set of ground truth targets.
            y_pred: The set of predicted targets.

        Returns: The loss of the given sets of ground truth and predicted targets.
        """
        raise NotImplementedError

    def derivative(self, y, y_pred):
        """Calculate the derivative of the loss function with respect to y.

        Arguments:
            y: The ground truth targets.
            y_pred: The predicted targets.

        Returns: The gradient of the loss function at the given values of `y` and `y_pred`.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class RMSE(Loss):
    """The RMSE (Root Mean Square Error) loss function."""

    def __call__(self, y, y_pred):
        error_term = y - y_pred
        self.grad = self.derivative(y, y_pred)

        return np.sqrt(np.mean(np.square(error_term), axis=0))

    def derivative(self, y, y_pred):
        return y_pred - y


class BinaryCrossEntropy(Loss):
    def __call__(self, y, y_pred):
        loss = -y * np.log(y_pred + Loss.EPSILON) - \
               (1 - y) * np.log(1 - y_pred + Loss.EPSILON)
        self.grad = self.derivative(y, y_pred)

        return loss

    def derivative(self, y, y_pred):
        return y * (y_pred - 1) + (1 - y) * y_pred


class CategoricalCrossEntropy(Loss):
    def __call__(self, y, y_pred):
        # Small value added to y_pred to avoid log(0), which is undefined.
        loss = -np.sum(y * np.log(y_pred + Loss.EPSILON), axis=1)

        self.grad = self.derivative(y, y_pred + Loss.EPSILON)

        return loss

    def derivative(self, y, y_pred):
        # This is the derivative of both cross entropy and the softmax
        # activation.
        return y_pred - y
