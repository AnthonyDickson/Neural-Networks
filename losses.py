"""This module implements a few loss functions that are used in neural
networks.
"""

import numpy as np


class Loss:
    """An abstraction of loss functions.

    A loss function provides two sets of functionality:
    1. Calculation of loss given a set of ground truths and predictions
    2. Storing of the gradient of the previously calculated loss.
    """

    def __init__(self):
        self.grad = None

    def __call__(self, y, y_pred):
        """Calculate loss.

        Arguments:
            y: The set of ground truth targets.
            y_pred: The set of predicted targets.

        Returns: The loss of the given sets of ground truth and predicted targets.
        """
        raise NotImplementedError


class RMSE(Loss):
    """The RMSE (Root Mean Square Error) loss function."""

    def __call__(self, y, y_pred):
        error_term = y - y_pred
        self.grad = error_term

        return np.sqrt(np.mean(np.square(error_term)))


# TODO: Implement cross-entropy loss.
class CrossEntropy(Loss):
    pass
