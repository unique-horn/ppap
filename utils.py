"""
Utility belt
"""

import numpy as np
import theano


def shared_matrix(size, low=-1, high=1):
    """
    Return a shared theano matrix with uniform value between the range

    Parameters
    ----------
    size : list_like
        Shape of the matrix
    low : float
        Lower limit for the matrix
    high : float
        Higher limit for the matrix

    Returns
    -------
    matrix : theano.shared
        Theano shared matrix
    """

    return theano.shared(np.random.uniform(size=size,
                                           low=low,
                                           high=high).astype(
                                               theano.config.floatX))
