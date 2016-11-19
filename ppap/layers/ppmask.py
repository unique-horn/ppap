"""
PPN based adaptive masking layers
"""

from keras.engine.topology import Layer

from .. import generators


class PPAdaptiveMask(Layer):
    """
    Return a generated mask
    To be used in conjugation with keras.layers.Merge and 'mul' mode
    """

    def __init__(self, output_shape, layer_sizes, **kwargs):
        """
        """

        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        self.gen = generators.FFMatrixGen(output_shape=output_shape,
                                          layer_sizes=layer_sizes)

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        """

        self.mask = self.gen.output

        self.trainable_weights = self.gen.weights + self.gen.biases
        self.non_trainable_weights = [self.mask]

        self.built = True

    def get_output_shape_for(self, input_shape):
        """
        """

        return self.output_shape

    def call(self, x, mask=None):
        """
        """

        return self.mask
