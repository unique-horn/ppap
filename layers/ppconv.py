"""
Convolutional layers with pattern producing networks
"""

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer


class PPConv(Layer):
    """
    Pattern producing convolutional layer
    Generates convolutional filter using a PPN
    """

    def __init__(self, **kwargs):
        super(PPConv, self).__init__(**kwargs)

    def build(self, input_dim):
        initial_weight_value = np.random.random((input_dim, 1))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
        self.built = True

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {"input_dim": self.input_dim}
        base_config = super(PPConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
