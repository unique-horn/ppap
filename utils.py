"""
Utility belt
"""

import numpy as np
import scipy.misc as smp
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


def show_image(image_data):
    """
    Display image from data

    Parameters
    ----------
    image_data : ndarray
    """

    img = smp.toimage(image_data)
    img.show()


def save_image(image_data, save_path):
    """
    Save image to given path
    """

    smp.toimage(image_data).save(save_path)
