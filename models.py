"""
Models
"""

import numpy as np
import theano
import theano.tensor as T

from utils import shared_matrix


class PPNGen(object):
    """
    Pattern Producing network class
    """

    def __init__(self, output_shape, layer_sizes):
        """
        Parameters
        ----------
        output_shape : list_like
            Size of the generated matrix (x, y)
        layer_sizes : array_like
            List of nodes in hidden layers
        """

        self.output_shape = output_shape
        self.layer_sizes = layer_sizes

        # List of weight matrices to learn
        self.weights = [shared_matrix((3, self.layer_sizes[0]))]

        for i in range(1, len(layer_sizes)):
            self.weights.append(shared_matrix((layer_sizes[i - 1], layer_sizes[
                i])))

        self.weights.append(shared_matrix((layer_sizes[-1], 1)))

        self.generator_function = self._generator()

    def _generator(self):
        """
        Return a theano function that generates data from unravelled
        coordinates
        """

        coordinates = T.matrix('coordinates')

        layer_outs = []

        layer_outs.append(theano.scan(lambda c: T.dot(c, self.weights[0]),
                                      sequences=coordinates)[0])

        for i in range(1, len(self.weights)):
            layer_outs.append(theano.scan(lambda c: T.dot(c, self.weights[i]),
                                          sequences=layer_outs[i - 1])[0])

        return theano.function(inputs=[coordinates], outputs=layer_outs[-1])

    def generate(self):
        """
        Generate an output matrix
        """

        x = np.arange(self.output_shape[0]) - self.output_shape[0] // 2
        y = np.arange(self.output_shape[1]) - self.output_shape[1] // 2
        x = x / x.max()
        y = y / y.max()

        # Generate coordinate data
        X, Y = np.meshgrid(x, y)
        R = (X**2) + (Y**2)

        total_values = np.prod(self.output_shape)

        # Unravelled
        Y_r = Y.reshape(total_values)
        X_r = X.reshape(total_values)
        R_r = R.reshape(total_values)

        vector_data = self.generator_function(np.vstack(
            [X_r, Y_r, R_r]).T.astype(theano.config.floatX))

        return vector_data.reshape(*self.output_shape)
