"""
This module contains mathematical tools for the INS implementation.
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np

def skew_symmetric_matrix(vector):
    """
    Returns the skew-symmetric matrix of a vector.
    """
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])
