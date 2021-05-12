import numpy as np 
from scipy.fftpack import dct


def mfcc(filter_banks, num_ceps=12, cep_lifter=22):
    """[summary]
    TODO
    Args:
        filter_banks ([type]): [description]
        num_ceps (int, optional): [description]. Defaults to 12.
        cep_lifter (int, optional): [description]. Defaults to 22.

    Returns:
        [type]: [description]
    """
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    return mfcc