import os
import sys

sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split

from mlp.activation_functions import Sigmoid, Identity
from mlp.datasets import load_iris
from mlp.layers import DenseLayer, GaussianNoise
from mlp.losses import RMSE
from mlp.network import MLPRegressor, EarlyStopping

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_iris(data_dir='../data/')
    X, y = utils.shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

    n_inputs = X.shape[1]
    mlp = MLPRegressor([GaussianNoise(n_inputs, n_inputs=n_inputs, std=0.01),
                        DenseLayer(n_inputs, activation_func=Sigmoid()),
                        DenseLayer(3, activation_func=Identity())],
                       learning_rate=0.03, momentum=0.9, loss_func=RMSE())

    train_loss, train_score, val_loss, val_score = mlp.fit(X_train, y_train, val_set=0.1, n_epochs=10000, batch_size=4,
                                                           shuffle_batches=True,
                                                           early_stopping=EarlyStopping(patience=400),
                                                           log_verbosity=100)
    print("Targets: %s - Predictions: %s" % (y_test.ravel(), mlp.predict(X_test).ravel()))
    print("Score: %.4f" % mlp.score(X_test, y_test))

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(train_score, label='train_score')
    plt.plot(val_score, label='val_score')
    plt.legend()
    plt.show()
