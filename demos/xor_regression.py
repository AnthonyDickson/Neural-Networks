import os
import sys

sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.activation_functions import Identity, LeakyReLU
from mlp.datasets import load_XOR
from mlp.layers import DenseLayer, GaussianNoise
from mlp.network import MLPRegressor, EarlyStopping

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_XOR(data_dir='../data/')
    X, y = utils.shuffle(X, y, random_state=42)

    mlp = MLPRegressor([GaussianNoise(2, n_inputs=2, std=0.01),
                        DenseLayer(2, activation_func=LeakyReLU()),
                        DenseLayer(1, activation_func=Identity())],
                       learning_rate=0.1)

    loss_history, _, _, _ = mlp.fit(X, y, n_epochs=10000, batch_size=-1, shuffle_batches=True,
                                    early_stopping=EarlyStopping(patience=100), log_verbosity=100)
    print("Targets: %s - Predictions: %s" % (y.ravel(), mlp.predict(X).ravel()))
    print("Score: %.4f" % mlp.score(X, y))

    plt.plot(loss_history)
    plt.show()
