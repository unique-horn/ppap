"""
PPN generators
"""
import numpy as np
from keras import backend as K
from keras import initializations


class FFMatrixGen:
    """
    Simple feed forward generator
    Doesn't take any explicit input
    """

    def __init__(self,
                 input_channels,
                 output_shape,
                 num_filters,
                 layer_sizes,
                 init="glorot_uniform"):
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
        self.input_channels = input_channels
        self.num_filters = num_filters
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
        l_sizes = [6] + self.layer_sizes + [1]

        self.weights = [self.init((l_sizes[i], l_sizes[i + 1]))
                        for i in range(len(l_sizes) - 1)]

        self.biases = [self.bias_init((b_size, ))
                       for index, b_size in enumerate(l_sizes[1:])]

    def setup_output(self):
        """
        Setup output tensor

        """

        coordinates = get_coordinates(self.output_shape,
                                      input_channels=self.input_channels,
                                      num_filters=self.num_filters)

        output = K.tanh(K.dot(coordinates, self.weights[0]) + self.biases[0])

        for i in range(1, len(self.weights) - 1):
            output = K.tanh(K.dot(output, self.weights[i]) + self.biases[i])
        output = K.sigmoid(K.dot(output, self.weights[-1]) + self.biases[-1])

        self.output = K.reshape(output, (self.num_filters, self.input_channels,
                                         *self.output_shape))


class FFMatrixGen2D:
    """
    NOTE: Vanilla FF matrix generator. Add separate generators without
    breaking this one.

    Simple feed forward generator
    """

    def __init__(self,
                 output_shape,
                 layer_sizes,
                 scale,
                 init="glorot_uniform"):
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
        self.scale = scale

        self.setup_weights()
        self.setup_output()

    def setup_weights(self):
        """
        Setup weights for the generator
        """

        # Layers with input and output
        l_sizes = [3] + self.layer_sizes + [1]

        self.weights = [self.init((l_sizes[i], l_sizes[i + 1]))
                        for i in range(len(l_sizes) - 1)]

        self.biases = [self.bias_init((b_size, ))
                       for b_size in enumerate(l_sizes[1:])]

    def setup_output(self):
        """
        Setup output tensor
        """

        coordinates = get_coordinates_2D(self.output_shape, scale=self.scale)

        output = K.sin(K.dot(coordinates, self.weights[0]) + self.biases[0])

        for i in range(1, len(self.weights) - 1):
            output = K.tanh(K.dot(output, self.weights[i]) + self.biases[i])
        output = K.sigmoid(K.dot(output, self.weights[-1]) + self.biases[-1])

        self.output = K.reshape(output, self.output_shape)


class FFGenZ:
    """
    Feed forward matrix generator which takes an input vector
    """

    def __init__(self,
                 output_shape,
                 z_dim,
                 layer_sizes,
                 scale,
                 init="glorot_uniform"):
        """
        Parameters
        ----------
        output_shape : list_like
            Size of the generated matrix (x, y)
        z_dim : int
            Size of the input z vector
        layer_sizes : list_like
            List of nodes in hidden layers
        scale : float
            Scale used for generating the coordinate matrix
            (see get_coordinates* functions)
        init : str
            Keras initializer to use for weights
        """

        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        self.z_dim = z_dim
        self.init = initializations.get(init)
        self.bias_init = initializations.get("zero")
        self.scale = scale

        self.setup_weights()
        self.setup_output()

    def setup_weights(self):
        """
        """

        l_sizes = [3] + self.layer_sizes + [1]

        self.weights = [self.init((l_sizes[i], l_sizes[i + 1]))
                        for i in range(len(l_sizes) - 1)]

        # Last term connects z vector to first hidden layer
        self.weights += [self.init((self.z_dim, self.layer_sizes[0]))]

        self.biases = [self.bias_init((b_size, )) for b_size in l_sizes[1:]]

    def setup_output(self):
        """
        """

        print("NOTE: Use get_output with input vector to get output")
        self.coordinates = get_coordinates_2D(self.output_shape,
                                              scale=self.scale)

    def get_output(self, z):
        """
        Return output using the given z
        z has shape (batch_size, z_dim)
        """

        assert len(z.shape) == 2
        assert self.z_dim == z.shape[1]

        total_values = np.prod(self.output_shape)
        batch_total = total_values * z.shape[0]

        z_rep = K.repeat_elements(K.expand_dims(z, 1), total_values, 1)

        coords_rep = K.repeat_elements(
            K.expand_dims(self.coordinates, 0), z.shape[0], 0)

        coords_rep = K.reshape(coords_rep,
                               (batch_total, self.coordinates.shape[1]))
        z_rep = K.reshape(z_rep, (batch_total, z.shape[1]))

        # Add z and coords to first layer
        output = K.sin(K.dot(coords_rep, self.weights[0]) + self.biases[0] +
                       K.dot(z_rep, self.weights[-1]))

        for i in range(1, len(self.layer_sizes)):
            output = K.tanh(K.dot(output, self.weights[i]) + self.biases[i])

        # Using -2 for weights since -1 is z vector weight
        output = K.sigmoid(K.dot(output, self.weights[-2]) + self.biases[-1])

        return K.reshape(output, (z.shape[0], *self.output_shape))


def get_coordinates_2D(matrix_shape, scale=5.0):
    """
    NOTE: Vanilla generator. Add different generators for different purpose
    rather than breaking this one.

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

    x = x / (x.max() + 1e-2)
    y = y / (y.max() + 1e-2)

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


def get_coordinates(matrix_shape, input_channels, num_filters, scale=1.0):
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
    c = np.arange(input_channels) - input_channels // 2
    f = np.arange(num_filters) - num_filters // 2

    x = x / (x.max() + 1)
    y = y / (y.max() + 1)
    c = c / (c.max() + 1)
    f = f / (f.max() + 1)  # to prevent division by zero

    x *= scale
    y *= scale
    c *= scale
    f *= scale
    # Generate coordinate data
    # the sequence in the meshgrid similar to output of generator
    F, C, X, Y = np.meshgrid(f, c, x, y)
    R = np.sqrt((X**2) + (Y**2) + (C**2) + (F**2))

    total_items = np.prod(matrix_shape) * num_filters * input_channels

    # Flatten
    Y_r = Y.reshape(total_items)
    X_r = X.reshape(total_items)
    C_r = C.reshape(total_items)
    F_r = F.reshape(total_items)
    R_r = R.reshape(total_items)

    # Random variable
    Rand = K.random_uniform_variable(shape=(Y_r.shape[0], 1), low=0, high=1)
    print(Rand.eval().shape)
    coordinates = K.variable(value=np.vstack([X_r, Y_r, C_r, F_r, R_r]).T)
    print(coordinates.eval().shape)
    coordinates = K.concatenate([Rand, coordinates], axis=1)
    return coordinates
