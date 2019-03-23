import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.activation_functions import Sigmoid, Softmax
from mlp.datasets import load_iris
from mlp.layers import DenseLayer
from mlp.losses import CategoricalCrossEntropy
from mlp.network import MLPClassifier

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_iris()
    X, y = utils.shuffle(X, y, random_state=42)

    n_inputs = X.shape[1]
    mlp = MLPClassifier([DenseLayer(n_inputs, n_inputs=n_inputs, activation_func=Sigmoid()),
                         DenseLayer(3, activation_func=Softmax())],
                        learning_rate=0.1,
                        momentum=0.9,
                        loss_func=CategoricalCrossEntropy())

    _, _, loss_history, _ = mlp.fit(X, y, n_epochs=10000, batch_size=4, early_stopping_patience=1000, log_verbosity=100,
                                    shuffle_batches=True)
    print("Targets: %s - Predictions: %s" % (y.ravel(), mlp.predict(X).ravel()))
    print("Score: %.4f" % mlp.score(X, y))

    plt.plot(loss_history)
    plt.show()
