"""
PPN generators
"""
# TODO: Generalized output with one more dimension.
import numpy as np
from keras import backend as K
from keras import initializations


class FFMatrixGen(object):
    """
    Simple feed forward generator
    Doesn't take any explicit input
    """

    def __init__(self, output_shape, layer_sizes, init="glorot_uniform"):
        """
        Parameters
        ----------
        output_shape : list_like
            Size of the generated matrix (x, y)
        layer_sizes : array_like
            List of nodes in hidden layers
        init : str
            Keras initializer to use for weights
        """

        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        self.init = initializations.get(init)
        self.bias_init = initializations.get("zero")

        self.setup_weights()
        self.setup_output()

    def setup_weights(self):
        """
        Setup weights for the generator
        """

        # Layers with input and output
        l_sizes = [3, *self.layer_sizes, 1]

        self.weights = [self.init((l_sizes[i], l_sizes[i + 1]))
                        for i in range(len(l_sizes) - 1)]

        self.biases = [self.bias_init((b_size, )) for b_size in l_sizes[1:]]

    def setup_output(self):
        """
        Setup output tensor

        """

        coordinates = get_coordinates(self.output_shape)

        output = K.sin(K.dot(coordinates, self.weights[0]) + self.biases[0])

        for i in range(1, len(self.weights) - 1):
            output.append(K.tanh(K.dot(output, self.weights[i]) + self.biases[
                i]))
        # The last might be wrong, as the i remains the same.
        output = K.sigmoid(K.dot(output, self.weights[-1]) + self.biases[i])

        self.output = K.reshape(output, (1, 1, *self.output_shape))


def get_coordinates(matrix_shape, scale=5.0):
    """
    Return meshgrid coordinates. Flattened and stacked in columns.

    Parameters
    ----------
    matrix_shape : list_like
        Shape of the output matrix
    scale : float
        Range of the coordinate representation (-scale, scale)

    Returns
    -------
    coords : keras tensor
    """

    # Generate coordinate data
    x = np.arange(matrix_shape[0]) - matrix_shape[0] // 2
    y = np.arange(matrix_shape[1]) - matrix_shape[1] // 2
    x = x / x.max()
    y = y / y.max()

    x *= scale
    y *= scale

    # Generate coordinate data
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X**2) + (Y**2))

    total_items = np.prod(matrix_shape)

    # Flatten
    Y_r = Y.reshape(total_items)
    X_r = X.reshape(total_items)
    R_r = R.reshape(total_items)

    return K.variable(value=np.vstack([X_r, Y_r, R_r]).T)
