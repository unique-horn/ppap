"""
PPN based adaptive masking layers
"""

import keras.backend as K
from keras import regularizers
from keras.engine.topology import Layer

from .. import generators


class PPAdaptiveMask(Layer):
    """
    Return a generated mask
    To be used in conjugation with keras.layers.Merge and 'mul' mode
    """

    def __init__(self, mask_shape, layer_sizes, scale, act_reg=None, **kwargs):
        """
        """

        self.mask_shape = mask_shape
        self.layer_sizes = layer_sizes
        self.scale = scale
        self.gen = generators.FFMatrixGen2D(output_shape=mask_shape,
                                            layer_sizes=layer_sizes,
                                            scale=scale)

        self.act_reg = regularizers.get(act_reg)

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        self.mask = self.gen.output

        self.trainable_weights = self.gen.weights + self.gen.biases
        self.non_trainable_weights = [self.mask]

        self.regularizers = []
        if self.act_reg:
            self.act_reg.set_layer(self)
            self.regularizers.append(self.act_reg)

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return (input_shape[0], 1, *self.mask_shape)

    def call(self, x, mask=None):
        """
        """

        return K.reshape(self.mask, [1, 1, *self.mask_shape])
