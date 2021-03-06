"""
Convolutional layers with pattern producing networks
"""
# TODO: Add names to the learnable paramters
# TODO The valid mode is not working
import keras.backend as K
from keras.engine.topology import Layer
from keras.utils.np_utils import conv_output_length
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
        self.layer_sizes = layer_sizes

        super().__init__(**kwargs)

    def build(self, input_dim):
        self.input_channels = input_dim[1]
        self.ppn_gen = generators.FFMatrixGen(output_shape=self.weight_shape,
                                              layer_sizes=self.layer_sizes,
                                              num_filters=self.nb_filters,
                                              input_channels=self.input_channels)
        self.W = self.ppn_gen.output  # PPN generator output, used as filter
        self.gen_weights = self.ppn_gen.weights # Weight of the generator
        self.gen_bias = self.ppn_gen.biases
        self.b = K.zeros((self.nb_filters))
        self.trainable_weights = self.gen_weights + self.gen_bias + [
            self.ppn_gen.z_r]
        self.non_trainable_weights = [self.W + self.b]
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