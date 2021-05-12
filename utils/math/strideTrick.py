import numpy as np


def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.
    Userfriendly version of creating sliding window views

    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.

    Returns:
        blocked/framed array.
    """
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, stride_length), strides=(stride_step*n, n))