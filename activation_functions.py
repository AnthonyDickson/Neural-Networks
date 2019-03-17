"""TODO: Add docstrings!"""

import numpy as np


class Acitvation:
    def __call__(self, X):
        raise NotImplementedError

    def forward(self, X):
        return self.__call__(X)

    def derivative(self, Y):
        raise NotImplementedError


class Sigmoid(Acitvation):
    def __call__(self, X):
        Z = np.exp(X)

        return Z / (Z + 1)

    def derivative(self, Y):
        return Y * (1 - Y)


class Identity(Acitvation):
    def __call__(self, X):
        return X

    def derivative(self, Y):
        return np.ones_like(Y)
