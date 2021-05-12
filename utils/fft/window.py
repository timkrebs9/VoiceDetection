import numpy as np


def hamming(frame_length, frames):
    """[summary]

    Args:
        frame_length (int): Numberof frames depending on the audio signal 
        frames (np array) : frames

    Returns:
        np array: returns framed signal with hamming window
    """
    frames *= np.hamming(frame_length)
    return frames
    #frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))