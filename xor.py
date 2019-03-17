import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from activation_functions import Sigmoid, Identity
from datasets import load_XOR
from mlp import MLP, Layer

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_XOR()
    X, y = utils.shuffle(X, y, random_state=42)
    mlp = MLP(0.1)
    mlp.add(Layer(2, 2, Sigmoid()))
    mlp.add(Layer(2, 1, Identity()))

    error_history = mlp.fit(X, y, n_epochs=10000, batch_size=1, early_stopping_threshold=100)
    print("Targets: %s - Predictions: %s" % (y.ravel(), mlp.predict_proba(X).ravel()))
    print("Score: %.4f (Lower is better)" % mlp.score(X, y))

    plt.plot(error_history)
    plt.show()
