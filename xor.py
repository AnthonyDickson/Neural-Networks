import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.activation_functions import Identity, ReLU
from mlp.datasets import load_XOR
from mlp.layers import DenseLayer
from mlp.network import MLPRegressor

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_XOR()
    X, y = utils.shuffle(X, y, random_state=42)

    mlp = MLPRegressor([DenseLayer(2, n_inputs=2, activation_func=ReLU()),
                        DenseLayer(1, activation_func=Identity())],
                       learning_rate=0.3)

    _, _, loss_history, _ = mlp.fit(X, y, n_epochs=10000, batch_size=-1,
                                    early_stopping_patience=100, log_verbosity=100, shuffle_batches=True)
    print("Targets: %s - Predictions: %s" % (y.ravel(), mlp.predict(X).ravel()))
    print("Score: %.4f" % mlp.score(X, y))

    plt.plot(loss_history)
    plt.show()
