import numpy as np
import math
from scipy.signal  import butter, lfilter
from scipy.special import jv


def movingAverage(x, sr, cutoff):
    """I
    mplementation of a Moving Average Filter (Low Pass)

    Args:
        x (np array): [  ] represents the aduio signal 
        sr (int )   : [Hz] sample rate sr = 2 * f_signal (Nyquist frequency)
        cutoff (int): [Hz] cutoff frequency refered to the 3 dB point 

    Returns:
        np array: filtered signal 
    """
    window = cutoff/ sr
    N = int(math.sqrt(0.196201 + window**2) / window)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def preemphasis(x, alpha):
    """
    Purpose of the Pre-Emphasis filter is to balance the frequency, 
    avoid numerical problems during the Fourier transform operation 
    and improve the Signal-to-Noise Ratio (SNR)

    y(t)=x(t)−αx(t−1)

    Args:
        x (np array) : represents the aduio signal
        alpha (float): filter coefficient (α)
                       default is between 0.95 - 0.97

    Returns:
        np array: filtered audio signal
    """
    x = np.append(x[0], x[1:] - alpha * x[:-1])
    return x



