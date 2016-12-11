"""
Implements the generator network proposed by HyperNetwork paper
"""
import keras.backend as K
from keras.engine.topology import Layer
from keras.utils.np_utils import conv_output_length
from .. import generators


class HyperNetwork(Layer):
    """
    Implements the generator network proposed by HyperNetwork paper
    """

    def __init__(self,
                 weight_shape,
                 hidden_dim,
                 strides=(1, 1),
                 border_mode="same",
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
        self.hidden_dim = hidden_dim

        super().__init__(**kwargs)

    def build(self, input_dim):
        self.input_channels = input_dim[1]
        self.ppn_gen = generators.HyperNetwork(output_shape=self.weight_shape,
                                               hidden_dim=self.hidden_dim,
                                              num_filters=self.nb_filters,
                                              input_channels=self.input_channels)
        self.W = self.ppn_gen.output  # PPN generator output, used as filter
        self.gen_weights = self.ppn_gen.weights # Weight of the generator
        self.gen_bias = self.ppn_gen.biases
        self.b = K.zeros((self.nb_filters))
        self.trainable_weights = self.gen_weights + self.gen_bias + [
            self.ppn_gen.z] + [self.b]
        self.non_trainable_weights = [self.W]
        self.built = True

    def call(self, x, mask=None):
        output = K.conv2d(x, self.W, border_mode=self.border_mode, strides=self.strides)
        output += K.reshape(self.b, (1, self.nb_filters, 1, 1))
        return output

    def get_output_shape_for(self, input_shape):

        rows = input_shape[2]
        cols = input_shape[3]

        rows = conv_output_length(rows, self.weight_shape[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.weight_shape[1],
                                  self.border_mode, self.strides[1])

        return (input_shape[0], self.nb_filters, rows, cols)