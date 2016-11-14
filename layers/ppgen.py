"""
Simple PPN based generative layers
"""

import numpy as np
from keras import backend as K
from keras import activations, initializations
from keras.engine.topology import Layer


class PPGenMatrix(Layer):
    """
    Pattern producing matrix generating layer
    Can be used to learn generating an image
    """

    def __init__(self,
                 matrix_shape,
                 layer_sizes,
                 scale=5.0,
                 init="glorot_uniform",
                 **kwargs):
        self.init = initializations.get(init)
        self.bias_init = initializations.get("zero")
        self.matrix_shape = matrix_shape
        self.layer_sizes = layer_sizes
        self.scale = scale
        super(PPGenMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        weights = [self.init((3, self.layer_sizes[0]), "W0")]
        biases = [self.bias_init((self.layer_sizes[0], ), "b0")]

        for i in range(1, len(self.layer_sizes)):
            weights.append(self.init(
                (self.layer_sizes[i - 1], self.layer_sizes[i]), "W" + str(i)))
            biases.append(self.bias_init(
                (self.layer_sizes[i], ), "b" + str(i)))

        weights.append(self.init((self.layer_sizes[-1], 1), "We"))
        biases.append(self.bias_init((1, ), "be"))

        self.Ws = weights
        self.bs = biases

        # Generate coordinate data
        x = np.arange(self.matrix_shape[0]) - self.matrix_shape[0] // 2
        y = np.arange(self.matrix_shape[1]) - self.matrix_shape[1] // 2
        x = x / x.max()
        y = y / y.max()

        x *= self.scale
        y *= self.scale

        # Generate coordinate data
        X, Y = np.meshgrid(x, y)
        R = (X**2) + (Y**2)

        total_values = np.prod(self.matrix_shape)

        # Unravelled
        Y_r = Y.reshape(total_values)
        X_r = X.reshape(total_values)
        R_r = R.reshape(total_values)

        self.coordinates = K.variable(value=np.vstack([X_r, Y_r, R_r]).T)

        self.trainable_weights = self.Ws + self.bs

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return (input_shape[0], *self.matrix_shape)

    def call(self, z, mask=None):
        """
        """

        output = K.tanh(K.dot(self.coordinates, self.Ws[0]) + self.bs[0])

        for i in range(1, len(self.layer_sizes)):
            output = K.tanh(K.dot(output, self.Ws[i]) + self.bs[i])

        output = K.sigmoid(K.dot(output, self.Ws[-1]) + self.bs[-1])

        a = K.reshape(output[:, 0], (self.matrix_shape))

        return K.expand_dims(a, 0)
