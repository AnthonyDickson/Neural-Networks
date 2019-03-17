import numpy as np


class Loss:
    def __init__(self):
        self.grad = None

    def __call__(self, y, y_pred):
        raise NotImplementedError


class RMSE(Loss):
    def __call__(self, y, y_pred):
        error_term = y - y_pred
        self.grad = error_term

        return np.sqrt(np.mean(np.square(error_term)))


# TODO: Implement cross-entropy loss.
class CrossEntropy(Loss):
    pass
