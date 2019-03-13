import numpy as np

from datasets import load_XOR
from mlp import MLP

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_XOR()
    mlp = MLP(X.shape[1], 2, 1, 1)

    mlp.fit(X, y, n_epochs=1000)
