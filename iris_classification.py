import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split

from mlp.activation_functions import Sigmoid, Softmax
from mlp.datasets import load_iris
from mlp.layers import DenseLayer
from mlp.losses import CategoricalCrossEntropy
from mlp.network import MLPClassifier, EarlyStopping

if __name__ == '__main__':
    np.random.seed(42)

    X, y = load_iris()
    X, y = utils.shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

    n_inputs = X.shape[1]
    mlp = MLPClassifier([DenseLayer(n_inputs, n_inputs=n_inputs, activation_func=Sigmoid()),
                         DenseLayer(3, activation_func=Softmax())],
                        learning_rate=0.1,
                        momentum=0.9,
                        loss_func=CategoricalCrossEntropy())

    es = EarlyStopping(patience=1000)

    train_loss, _, val_loss, _ = mlp.fit(X_train, y_train, val_set=0.1, n_epochs=10000, batch_size=4,
                                         shuffle_batches=True,
                                         early_stopping=es, log_verbosity=100)
    print("Targets: %s - Predictions: %s" % (y.ravel(), mlp.predict(X).ravel()))
    print("Score: %.4f" % mlp.score(X, y))

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.show()
