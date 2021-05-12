import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils.frame.framing import framing



def normalizedShortTimeEnergy(frames):
    return np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames)))**2, axis=-1) / len(frames)**2


def naive_frame_energy_vad(sig, fs, threshold=-20, win_len=0.25, win_hop=0.25, E0=1e7):
    """
    Calculate the short time energy. The short time energy can be assumed as the total energy of each frame 

    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
        threshold (int ): Energy threshold
                          Default is -20 [dB]
        win_len (float) : window length in sec.
                          Default is 0.25 ms.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.25.
        E0 ([type]): Energy of the Signal
                          Default is 1e7.

    Returns:
        [type]: [description]
    """
    # framing
    frames, frames_len = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # compute short time energies to get voiced frames
    energy = normalizedShortTimeEnergy(frames)
    log_energy = 10 * np.log10(energy / E0)
    # normalize energy to 0 dB then filter and format
    energy = scipy.signal.medfilt(log_energy, 5)
    energy = np.repeat(energy, frames_len)

    # compute vad and get speech frames
    vad     = np.array(energy > threshold, dtype=sig.dtype)
    vframes = np.array(frames.flatten()[np.where(vad==1)], dtype=sig.dtype)
    return energy, vad, np.array(vframes, dtype=np.float64)



def fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(n/2))
    
    plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequenz")
    plt.ylabel("Magnitude")
    return plt.show()

def fft2(audio, samplerate):
    n = len(audio)
    T = 1/samplerate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(n/2))
    return yf



def powerfft(frames, NFFT=512):
    """
    N -point FFT on each frame to calculate the frequency spectrum, which is also called 
    Short-Time Fourier-Transform (STFT)
        P=|FFT(xi)|^2 / N
    where, xi is the ith frame of signal x
    
    Args:
        frames (np array): framed signal
        NFFT (int)       : Number of N points of the fft
                           default is 256 or 512

    Returns:
        np array: returns power spectrum of the framed signal
    """
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    power_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return power_frames
