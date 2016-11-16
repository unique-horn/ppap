"""
Simple PPN based generative layers
"""

import numpy as np
from keras import backend as K
from keras import initializations
from keras.engine.topology import Layer

from .. import generators


class PPGenMatrix(Layer):
    """
    Pattern producing matrix generating layer
    Can be used to learn generating an image
    """

    def __init__(self,
                 matrix_shape,
                 layer_sizes,
                 z_dim,
                 scale=5.0,
                 init="glorot_uniform",
                 **kwargs):
        """
        Parameters
        ----------
        matrix_shape : list_like
            Shape of output
        layer_sizes : list_like
            List of number of hidden nodes in layers of generator
        z_dim : int
            Dimension of input vector
        scale : float
            Range of internal coordinate representation (-scale, scale)
        init : str
            Keras initializer to use for generator weights (not bias)
        """

        self.init = initializations.get(init)
        self.bias_init = initializations.get("zero")
        self.matrix_shape = matrix_shape
        self.layer_sizes = layer_sizes
        self.input_dim = z_dim
        self.scale = scale

        kwargs["input_shape"] = (self.input_dim, )
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        weight_z = self.init((input_shape[1], self.layer_sizes[0]), "Wz")

        weights = [self.init((3, self.layer_sizes[0]), "W0")]
        biases = [self.bias_init((self.layer_sizes[0], ), "b0")]

        for i in range(1, len(self.layer_sizes)):
            weights.append(self.init(
                (self.layer_sizes[i - 1], self.layer_sizes[i]), "W" + str(i)))
            biases.append(self.bias_init(
                (self.layer_sizes[i], ), "b" + str(i)))

        weights.append(self.init((self.layer_sizes[-1], 1), "We"))
        biases.append(self.bias_init((1, ), "be"))

        self.Wz = weight_z
        self.Ws = weights
        self.bs = biases

        self.coordinates = generators.get_coordinates(self.matrix_shape,
                                                      scale=self.scale)

        self.trainable_weights = self.Ws + self.bs + [self.Wz]

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return (input_shape[0], *self.matrix_shape)

    def call(self, z, mask=None):
        """
        """

        total_values = np.prod(self.matrix_shape)
        batch_total = total_values * z.shape[0]

        # Expand z on pixel dimension
        z_rep = K.repeat_elements(
            K.expand_dims(z, 1), total_values, 1) * self.scale

        # Expand coordinates on batch dimension
        coords_rep = K.repeat_elements(
            K.expand_dims(self.coordinates, 0), z.shape[0], 0)

        # Merge top two dimensions
        coords_rep = K.reshape(coords_rep,
                               (batch_total, self.coordinates.shape[1]))
        z_rep = K.reshape(z_rep, (batch_total, z.shape[1]))

        # Add z and coords to first layer
        output = K.sin(K.dot(coords_rep, self.Ws[0]) + self.bs[0] + K.dot(
            z_rep, self.Wz))

        for i in range(1, len(self.layer_sizes)):
            output = K.tanh(K.dot(output, self.Ws[i]) + self.bs[i])

        output = K.sigmoid(K.dot(output, self.Ws[-1]) + self.bs[-1])

        return K.reshape(output, (z.shape[0], *self.matrix_shape))
