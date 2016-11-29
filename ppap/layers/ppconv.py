"""
Convolutional layers with pattern producing networks
"""

import keras.backend as K
from keras.engine.topology import Layer

from .. import generators


class PPConv(Layer):
    """
    Pattern producing convolutional layer
    Generates convolutional filter using a PPN
    """

    def __init__(self,
                 weight_shape,
                 layer_sizes,
                 strides=(1, 1),
                 border_mode="valid",
                 nb_filters=1,
                 dim_ordering="th",
                 **kwargs):
        """

        :param weight_shape:
        :param layer_sizes:
        """

        self.strides = strides
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.weight_shape = weight_shape
        self.nb_filters = nb_filters
        self.ppn_gen = generators.FFMatrixGen(output_shape=weight_shape,
                                              layer_sizes=layer_sizes)

        super().__init__(**kwargs)

    def build(self, input_dim):
        self.W = self.ppn_gen.output  # PPN generator output, used as filter
        self.gen_weights = self.ppn_gen.weights + self.ppn_gen.biases  # Weight of the generator

        self.b = K.zeros((self.nb_filters))
        self.trainable_weights = [self.gen_weights]
        self.non_trainable_weights = [self.W + self.b]

        self.built = True

    def call(self, x, mask=None):
        output = K.conv2d(x, self.W, border_mode="same", strides=self.strides)
        output += K.reshape(self.b, (1, self.nb_filters, 1, 1))
        return output

    def get_output_shape_for(self, input_shape):
        # length = conv_output_length(input_shape[1],
        #                             self.weight_shape[0],
        #                             self.border_mode,
        #                             self.strides[0])
        print(input_shape)
        # print (input_shape[0], length, self.nb_filters)
        if self.nb_filters >= 1:
            return (input_shape)
        else:
            return (input_shape)