"""
Dynamic filtering layers using PPNs
"""

import keras.backend as K
from keras.engine.topology import Layer
from keras.utils.np_utils import conv_output_length

from .. import generators


class PPDFN(Layer):
    """
    Basic Dynamic filtering layer.
    Takes input of shape (batch_size, channels, rows, columns)
    and returns output of shape (batch_size, filters, rows, columns)
    ; 'th' ordering noted
    """

    def __init__(self, n_filters, n_rows, n_cols, **kwargs):
        """
        """

        # Use set_image_dim_ordering global to control this
        self.dim_ordering = K.image_dim_ordering()
        self.n_filters = n_filters
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.border_mode = "same"
        self.strides = (1, 1)

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        batch_size, self.filters_in = input_shape[:2]

        if self.dim_ordering == "th":
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == "tf":
            rows = input_shape[1]
            cols = input_shape[2]

        self.gen = generators.PPDFGen(self.n_filters, (rows, cols))
        self.trainable_weights = self.gen.weights + self.gen.biases

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        if self.dim_ordering == "th":
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == "tf":
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_output_length(rows, self.nb_row, self.border_mode,
                                  self.strides[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode,
                                  self.strides[1])

        if self.dim_ordering == "th":
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == "tf":
            return (input_shape[0], rows, cols, self.nb_filter)

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

        filters = self.gen.get_output(x)
        pass
