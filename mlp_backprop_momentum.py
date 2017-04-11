import numpy as np


class MLP:
    """
    This code was adapted from:
    https://rolisz.ro/2013/04/18/neural-networks-in-python/
    """

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_derivative(self, a):
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        return a * (1.0 - a)

    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]
        self.layers = layers
        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_derivative

        self.init_weights()

    def init_weights(self):
        self.weights = []
        self.delta_weights = []
        for i in range(1, len(self.layers) - 1):
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) - 1) * 0.25)
            self.delta_weights.append(np.zeros((self.layers[i - 1] + 1, self.layers[i] + 1)))
        self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)
        self.delta_weights.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))

    def fit(self, data_train, data_test=None, learning_rate=0.1, momentum=0.0, epochs=100):
        """
        Online learning.
        :param data_train: training data, tuple (input, output)
        :param data_test: test data, tuple (input, output)
        :param learning_rate: learning speed
        :param momentum: learning inertia
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.atleast_2d(data_train[0])
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(data_train[1])
        error_train = np.zeros(epochs)
        if data_test is not None:
            error_test = np.zeros(epochs)
            out_test = np.zeros(data_test[1].shape)

        a = []
        for l in self.layers:
            a.append(np.zeros(l))

        for k in range(epochs):
            error_it = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                a[0] = X[i]

                for l in range(len(self.weights)):
                    a[l + 1] = self.activation(np.dot(a[l], self.weights[l]))

                error = a[-1] - y[i]
                error_it[it] = np.mean(error ** 2)
                deltas = [error * self.activation_deriv(a[-1])]

                # we need to begin at the layer previous to the last one
                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.delta_weights[i] = (-learning_rate * layer.T.dot(delta)) + (momentum * self.delta_weights[i])
                    self.weights[i] += self.delta_weights[i]

            error_train[k] = np.mean(error_it)
            if data_test is not None:
                error_test[k], _ = self.compute_MSE(data_test)

        if data_test is None:
            return error_train
        else:
            return (error_train, error_test)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def compute_output(self, data):
        assert len(data[1].shape) == 2, "data[1] must be a 2-dimensional array"

        out = np.zeros(data[1].shape)
        for r in np.arange(data[0].shape[0]):
            out[r, :] = self.predict(data[0][r, :])
        return out

    def compute_MSE(self, data):
        assert len(data[1].shape) == 2, "data[1] must be a 2-dimensional array"

        out = self.compute_output(data)
        return (np.mean((data[1] - out) ** 2), out)
