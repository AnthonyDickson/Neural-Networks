import numpy as np
from sklearn.utils import shuffle

from datasets import load_XOR
from mlp import MLP

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_XOR()
    X, y = shuffle(X, y, random_state=42)
    mlp = MLP(X.shape[1], 2, 1, 2)

    mlp.fit(X, y, n_epochs=2000, batch_size=4)
    print("Score: %.4f (Lower is better)" % mlp.score(X, y))
