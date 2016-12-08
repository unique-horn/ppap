"""
Simple PPN based generative layers
"""

from keras.engine.topology import Layer

from .. import generators


class PPGenMatrix(Layer):
    """
    Pattern producing matrix generating layer
    Can be used to learn generating an image
    """

    def __init__(self, matrix_shape, layer_sizes, z_dim, scale=5.0, **kwargs):
        """
        Parameters
        ----------
        matrix_shape : list_like
            Shape of output
        layer_sizes : list_like
            List of number of hidden nodes in layers of generator
        z_dim : int
            Dimension of input vector
        scale : float
            Range of internal coordinate representation (-scale, scale)
        """

        self.matrix_shape = matrix_shape
        self.layer_sizes = layer_sizes
        self.input_dim = z_dim
        self.scale = scale

        kwargs["input_shape"] = (self.input_dim, )
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        self.gen = generators.FFGenZ(self.matrix_shape, self.input_dim,
                                     self.layer_sizes, self.scale)

        self.trainable_weights = self.gen.weights + self.gen.biases

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return (input_shape[0], *self.matrix_shape)

    def call(self, z, mask=None):
        """
        """

        return self.gen.get_output(z)
