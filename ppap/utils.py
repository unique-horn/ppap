"""
Utilities
"""

from noise import pnoise3
import numpy as np


def perlin_batch(batch_size, rows, cols, perlin_spans, coordinate_span):
    """
    Generate coordinates with perlin noise
    """

    perlin = np.zeros((batch_size, rows, cols))

    for i in np.arange(batch_size):
        for j in np.arange(rows):
            for k in np.arange(cols):
                perlin[i, j, k] = pnoise3(
                    float(i) * perlin_spans[0] / batch_size - 0.5 *
                    perlin_spans[0],
                    float(j) * perlin_spans[1] / rows - 0.5 * perlin_spans[1],
                    float(k) * perlin_spans[2] / cols - 0.5 * perlin_spans[2])

    x = np.arange(rows) - rows // 2
    y = np.arange(cols) - cols // 2

    x = x / (x.max() + np.finfo(float).eps)
    y = y / (y.max() + np.finfo(float).eps)

    x *= coordinate_span
    y *= coordinate_span
    # Generate coordinate data
    X, Y = np.meshgrid(x, y)

    X = X + perlin
    Y = Y + perlin
    R = np.sqrt((X**2) + (Y**2))

    return X, Y, R
