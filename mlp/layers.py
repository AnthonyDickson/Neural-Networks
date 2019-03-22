import numpy as np

from mlp.activation_functions import Identity


class DenseLayer:
    """A fully connected layer in a neural network."""

    def __init__(self, n_units, n_inputs=None, activation_func=None):
        """Create a fully connected layer.

        Arguments;
            n_units: How many units (or artificial neurons) the layer should have.

            n_inputs: How many inputs the layer should accept. If this is set to
            None then the input size will be inferred from the previous layer.

            activation_func: The activation function to use for this layer.
            If set the None this defaults to the identity function.
        """
        self.n_inputs = n_inputs
        self.n_units = n_units

        self.W = None
        self.b = None
        self.prev_dW = None
        self.prev_db = None

        self.prev_input = None
        self.activation_value = None
        self.preactivation_value = None

        self.activation_func = activation_func if activation_func is not None else Identity()

        self.is_output = False
        self.network = None
        self.next_layer = None

        self.initialised = False

    def initialise_weights(self):
        """Create and initialise the weight and bias matrices."""
        self.W = np.random.normal(0, 1, (self.n_inputs, self.n_units)) * np.sqrt(1.0 / self.n_inputs)
        self.b = np.random.normal(0, 1, (1, self.n_units))
        self.prev_dW = np.zeros_like(self.W)
        self.prev_db = np.zeros_like(self.b)
        self.initialised = True

    def clone(self, deep=False):
        """Create a clone of a layer.

        Cached activations, references to the parent network and the next layer etc. are not copied.

        Arguments:
            deep: If set to True a deep copy is created, otherwise only a `topological` copy is created,
            i.e. weights are not copied.

        Returns: A copy of the layer.
        """
        if deep:
            layer = DenseLayer(self.n_units, self.n_inputs, self.activation_func.clone())
            layer.W = self.W.copy()
            layer.b = self.b.copy()
            layer.initialised = self.initialised

            return layer
        else:
            return DenseLayer(self.n_units, self.n_inputs, self.activation_func)

    @property
    def shape(self):
        """Get the shape of the layer,

        Returns: A 2-tuple containing the number of inputs for each unit and
        the number of units in the layer.
        """
        return self.W.shape

    def forward(self, X):
        """Perform a forward pass of the MLP.

        Returns: The activation of the layer.
        """
        self.prev_input = X

        output = np.matmul(X, self.W) + self.b
        self.preactivation_value = output

        output = self.activation_func(output)
        self.activation_value = output

        return output

    def backward(self, error_term):
        """Perform a backward pass of the layer (i.e. back propagate error) and
        update weights and biases.

        Arguments:
            error_term: The error term for the output layer, or the deltas of
            the previous layer in the case of a hidden layer.

        Returns the calculated deltas for the given layer.
        """
        N = error_term.shape[0]

        if self.is_output:
            delta = error_term
        else:
            delta = error_term.dot(self.next_layer.W.T)

        delta *= self.activation_func.derivative(self.activation_value)

        dW = self.network.learning_rate * np.matmul(self.prev_input.T, delta) \
             + self.network.momentum * self.prev_dW
        db = self.network.learning_rate * delta + \
             self.network.momentum * self.prev_db

        dW_mean = dW / N
        db_mean = db.mean(axis=0)

        self.W -= dW_mean
        self.b -= db_mean

        self.prev_dW = dW_mean
        self.prev_db = db_mean

        return delta

    def __str__(self):
        return '%s(n_inputs=%d, n_units=%d, activation_func=%s())' \
               % (self.__class__.__name__, self.n_inputs, self.n_units, str(self.activation_func))
