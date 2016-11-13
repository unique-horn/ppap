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
                 init="glorot_uniform",
                 **kwargs):
        self.init = initializations.get(init)
        self.matrix_shape = matrix_shape
        self.layer_sizes = layer_sizes
        super(PPGenMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        weights = [self.init((3, self.layer_sizes[0]), name="in")]
        for i in range(1, len(self.layer_sizes)):
            weights.append(self.init(
                (self.layer_sizes[i - 1], self.layer_sizes[i]),
                name="weight" + str(i)))
        weights.append(self.init((self.layer_sizes[-1], 1), name="out"))

        # Generate coordinate data
        x = np.arange(self.matrix_shape[0]) - self.matrix_shape[0] // 2
        y = np.arange(self.matrix_shape[1]) - self.matrix_shape[1] // 2
        x = x / x.max()
        y = y / y.max()

        # Generate coordinate data
        X, Y = np.meshgrid(x, y)
        R = (X**2) + (Y**2)

        total_values = np.prod(self.matrix_shape)

        # Unravelled
        Y_r = Y.reshape(total_values)
        X_r = X.reshape(total_values)
        R_r = R.reshape(total_values)

        self.coordinates = K.variable(value=np.vstack([X_r, Y_r, R_r]).T)

        self.trainable_weights = weights

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return (input_shape[0], *self.matrix_shape)

    def call(self, z, mask=None):
        """
        """

        layer_outs = [K.dot(self.coordinates, self.weights[0])]

        for i in range(1, len(self.weights)):
            layer_outs.append(K.dot(layer_outs[i - 1], self.weights[i]))

        a = K.reshape(layer_outs[-1][:, 0], (self.matrix_shape))

        return K.expand_dims(a, 0)
