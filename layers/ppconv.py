"""
Convolutional layers with pattern producing networks
"""
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from experiments.models import PPNGen
from keras.utils.np_utils import conv_output_length
from keras.layers import Convolution2D
# TODO: Initialization of the weights in good way
# TODO: Inheriting the Convolutional2D layer for the PPConv Layer
#
class PPConv(Layer):
    """
    Pattern producing convolutional layer
    Generates convolutional filter using a PPN
    """

    def __init__(self, weight_shape, layer_sizes, strides=(1, 1),
                 border_mode='valid', nb_filters=1, dim_ordering="th",
                 **kwargs):
        """

        :param weight_shape:
        :param layer_sizes:
        """
        super(PPConv, self).__init__(**kwargs)
        self.strides = strides
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.weight_shape = weight_shape
        self.nb_filters = nb_filters
        self.ppn_gen = PPNGen(output_shape=weight_shape,
                             layer_sizes=layer_sizes)

    def build(self, input_dim):
        Weights, gen_weights = self.ppn_gen._generator()
        self.W = Weights
        self.gen_weights = gen_weights
        self.b = K.zeros((self.nb_filters))
        self.trainable_weights = self.gen_weights + [self.b]
        self.non_trainable_weights = self.W
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
        print (input_shape)
        # print (input_shape[0], length, self.nb_filters)
        if self.nb_filters >= 1:
            return (input_shape)
        else:
            return (input_shape)