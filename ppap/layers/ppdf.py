"""
Dynamic filtering layers using PPNs
"""

import keras.backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.utils.np_utils import conv_output_length

from .. import generators


class PPDFN(Layer):
    """
    PP Dynamic filtering layer.
    Takes input of shape (batch_size, channels, rows, columns)
    and returns output of shape (batch_size, channels, rows, columns)
    ; 'th' ordering noted
    """

    def __init__(self, filter_size, **kwargs):
        """
        Parameters:
        -----------
        filter_size : int
            Size of filter along 1 dimension
        """

        # Use set_image_dim_ordering global to control this
        self.dim_ordering = K.image_dim_ordering()
        self.filter_size = filter_size
        self.border_mode = "same"
        self.strides = (1, 1)

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        self.batch_size = input_shape[0]

        if self.dim_ordering == "th":
            self.n_rows = input_shape[2]
            self.n_cols = input_shape[3]
            self.filters_in = input_shape[1]
        elif self.dim_ordering == "tf":
            self.n_rows = input_shape[1]
            self.n_cols = input_shape[2]
            self.filters_in = input_shape[3]

        self.gen = generators.PPDFGen(self.filter_size,
                                      (self.n_rows, self.n_cols),
                                      self.filters_in, self.batch_size)
        self.trainable_weights = self.gen.weights + self.gen.biases

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        rows = conv_output_length(self.n_rows, self.filter_size,
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(self.n_cols, self.filter_size,
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == "th":
            return (input_shape[0], self.filters_in, rows, cols)
        elif self.dim_ordering == "tf":
            return (input_shape[0], rows, cols, self.filters_in)

    def call(self, x, mask=None):
        """
        """

        # Assuming 'th' ordering
        # Input shape is:
        #   (batch, channels, rows, cols) ; e.g. (20, 3, 32, 32)
        # Generate a filter of size, incorporate the coordinates here
        #   (batch, filter_size ** 2, rows, cols) ; (20, 7*7, 32, 32)
        # Expand each channel to (filter_size ** 2) by shifting it using the
        # dfn trick
        # Do an elementwise dot with filter and dimension reduction on channel
        # axis

        # Generated filter tensors
        filters = self.gen.get_output(x)

        # Alias
        fs = self.filter_size
        # Aux convolution to shift the images
        if self.dim_ordering == "th":
            shifter_shape = (fs**2, 1, fs, fs)
            ch_axis = 1
        elif self.dim_ordering == "tf":
            shifter_shape = (fs**2, fs, fs, 1)
            ch_axis = 3

        shifter = np.reshape(np.eye(fs**2, fs**2), shifter_shape)

        shifter = K.variable(value=shifter)

        # Use same filter in all channels and return same number of channels
        outputs = []
        for i in range(self.filters_in):
            if self.dim_ordering == "th":
                x_channel = x[:, [i], :, :]
            elif self.dim_ordering == "tf":
                x_channel = x[:, :, :, [i]]

            # This creates shifted version of x in all direction
            # When stacked together and summed after elemwise mult with
            # filters, this results in an effective convolution
            x_shifted = K.conv2d(
                x_channel,
                shifter,
                strides=self.strides,
                border_mode=self.border_mode,
                dim_ordering=self.dim_ordering)

            output = K.sum(x_shifted * filters, axis=ch_axis, keepdims=True)
            outputs.append(output)

        output = K.concatenate(outputs, axis=ch_axis)

        return output


class DFN(Layer):
    """
    Dynamic filtering layer.
    Takes input of shape (batch_size, channels, rows, columns)
    and returns output of shape (batch_size, channels, rows, columns)
    ; 'th' ordering noted
    """

    def __init__(self, filter_size, **kwargs):
        """
        Parameters:
        -----------
        filter_size : int
            Size of filter along 1 dimension
        """

        # Use set_image_dim_ordering global to control this
        self.dim_ordering = K.image_dim_ordering()
        self.filter_size = filter_size
        self.border_mode = "same"
        self.strides = (1, 1)

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        self.batch_size = input_shape[0]

        if self.dim_ordering == "th":
            self.n_rows = input_shape[2]
            self.n_cols = input_shape[3]
            self.filters_in = input_shape[1]
        elif self.dim_ordering == "tf":
            self.n_rows = input_shape[1]
            self.n_cols = input_shape[2]
            self.filters_in = input_shape[3]

        self.gen = generators.DFGen(self.filter_size,
                                    (self.n_rows, self.n_cols),
                                    self.filters_in, self.batch_size)
        self.trainable_weights = self.gen.weights + self.gen.biases

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        rows = conv_output_length(self.n_rows, self.filter_size,
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(self.n_cols, self.filter_size,
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == "th":
            return (input_shape[0], self.filters_in, rows, cols)
        elif self.dim_ordering == "tf":
            return (input_shape[0], rows, cols, self.filters_in)

    def call(self, x, mask=None):
        """
        """

        # Assuming 'th' ordering
        # Input shape is:
        #   (batch, channels, rows, cols) ; e.g. (20, 3, 32, 32)
        # Generate a filter of size, incorporate the coordinates here
        #   (batch, filter_size ** 2, rows, cols) ; (20, 7*7, 32, 32)
        # Expand each channel to (filter_size ** 2) by shifting it using the
        # dfn trick
        # Do an elementwise dot with filter and dimension reduction on channel
        # axis

        # Generated filter tensors
        filters = self.gen.get_output(x)

        # Alias
        fs = self.filter_size
        # Aux convolution to shift the images
        if self.dim_ordering == "th":
            shifter_shape = (fs**2, 1, fs, fs)
            ch_axis = 1
        elif self.dim_ordering == "tf":
            shifter_shape = (fs**2, fs, fs, 1)
            ch_axis = 3

        shifter = np.reshape(np.eye(fs**2, fs**2), shifter_shape)

        shifter = K.variable(value=shifter)

        # Use same filter in all channels and return same number of channels
        outputs = []
        for i in range(self.filters_in):
            if self.dim_ordering == "th":
                x_channel = x[:, [i], :, :]
            elif self.dim_ordering == "tf":
                x_channel = x[:, :, :, [i]]

            # This creates shifted version of x in all direction
            # When stacked together and summed after elemwise mult with
            # filters, this results in an effective convolution
            x_shifted = K.conv2d(
                x_channel,
                shifter,
                strides=self.strides,
                border_mode=self.border_mode,
                dim_ordering=self.dim_ordering)

            output = K.sum(x_shifted * filters, axis=ch_axis, keepdims=True)
            outputs.append(output)

        output = K.concatenate(outputs, axis=ch_axis)

        return output
