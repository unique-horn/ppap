"""
This is an extension of the HyperNetwork, where the layer specific input z's
will be conditioned on the input. We first find the max of the input across
the feature channel, then we project this reduced input to the dimension of
z's, which is 4 here.
"""
import keras.backend as K
from keras.engine.topology import Layer
from keras.utils.np_utils import conv_output_length
from keras import initializations
import numpy as np

class HyperNetwork_max(Layer):
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
    rows, cols = input_dim[2], input_dim[3]

    self.ppn_gen = HyperNetwork_gen(output_shape=self.weight_shape,
                                           rows=rows, cols=cols,
                                           hidden_dim=self.hidden_dim,
                                           num_filters=self.nb_filters,
                                           input_channels=self.input_channels)
    self.gen_weights = self.ppn_gen.weights  # Weight of the generator
    self.gen_bias = self.ppn_gen.biases
    self.b = K.zeros((self.nb_filters))
    self.trainable_weights = self.gen_weights + self.gen_bias + [self.b]
    self.built = True

  def call(self, x, mask=None):
    self.W = self.ppn_gen.setup_output(x)  # PPN generator output, used as filter
    # self.non_trainable_weights = [self.W]
    output = K.conv2d(x, self.W, border_mode=self.border_mode,
                      strides=self.strides)
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


class HyperNetwork_gen(object):
  """
  Simple feed forward generator
  Doesn't take any explicit input
  """

  def __init__(self,
               input_channels,
               rows, cols,
               output_shape,
               num_filters,
               hidden_dim,
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
    self.input_rows = rows
    self.input_cols = cols
    self.input_channels = input_channels
    self.num_filters = num_filters
    self.output_shape = output_shape
    self.hidden_dim = hidden_dim
    self.init = initializations.get(init)
    self.bias_init = initializations.get("zero")
    self.setup_weights()
    self.num_param = np.prod(self.output_shape) * self.num_filters * \
                     self.input_channels

  def setup_weights(self):
    """
    Setup weights for the generator
    """
    # Layers with input and output
    self.w_proj_to_z = self.init((self.input_cols * self.input_rows, 4))
    # self.b_proj_to_z = self.bias_init((4))

    w1 = self.init((4, self.hidden_dim * self.input_channels))
    b1 = self.bias_init((self.hidden_dim * self.input_channels))

    w2 = self.init((self.hidden_dim, np.prod(self.output_shape) *
                    self.num_filters))  # (hid X 3*3*33)
    b2 = self.bias_init((np.prod(self.output_shape) *
                         self.num_filters))
    # self.z = self.init((1, 4))
    self.weights = [w1, w2, self.w_proj_to_z]
    self.biases = [b1, b2]

  def setup_output(self, x):
    """
    Setup output tensor

    """
    x_max = K.max(x, axis=1)

    x_max = K.flatten(x_max)
    z = K.dot(x_max, self.w_proj_to_z) #+ self.b_proj_to_z
    hidden = K.dot(z, self.weights[0]) + self.biases[0]
    hidden = K.reshape(hidden, shape=(self.input_channels,
                                      self.hidden_dim))

    output = K.dot(hidden, self.weights[1]) + self.biases[1]

    self.output = K.reshape(output, (self.num_filters, self.input_channels,
                                     *self.output_shape))
    return self.output