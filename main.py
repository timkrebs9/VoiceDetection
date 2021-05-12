import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from utils.filt.filter import *
from utils.fft.fft import naive_frame_energy_vad
from utils.math.statistica import Statistics
from plot.visualizer import multi_plots


# 1. Step: Vorverarbeitung
#   1.1 Abtastung
#   1.2 Filterung
#   1.3 Trasnformation
#   1.4 Merkmalsvektor
#   1.5 Cepstrum 

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio





if __name__ == '__main__':

    filename = 'OSR_us_000_0060_8k.wav'

    # Read audio file in array
    fs, sig = scipy.io.wavfile.read("OSR/"+filename)
   


    # Filter the Signal with Low Pass Filter
        #sig = movingAverage(x=sig, sr=fs, cutoff=8000)
    sig = preemphasis(x=sig, alpha= 0.97)



    # get voiced frames
    energy, vad, voiced = naive_frame_energy_vad(sig, fs, threshold=-20, win_len=0.025, win_hop=0.025)
    
    
    # plot results
        #multi_plots(data=[sig, energy, vad, voiced], titles=["Input signal (voiced + silence)", "Short time energy", "Voice activity detection", "Output signal (voiced only)"], fs=fs, plot_rows=4, step=1)

    # save voiced signal
        #scipy.io.wavfile.write("rame_energy_vad"+ filename, fs,  np.array(voiced, dtype=sig.dtype))

#########
# Test
########
    voiced = preemphasis(x=voiced, alpha= 0.97)

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * fs, frame_stride * fs  # Convert from seconds to samples
    signal_length = len(voiced)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(voiced, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]


    # Windowing
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks
    nfilt = 40      # Number of Filters
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB


    # Mel-frequency Cepstral Coefficients (MFCCs)
    num_ceps = 12
    cep_lifter = 22
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape

    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*


    # Main Normalization
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    plt.imshow(mfcc.T,  cmap=plt.cm.jet, aspect='auto')
    plt.title('Filter Bank')
    plt.xticks(np.arange(0, (filter_banks.T).shape[1], int((filter_banks.T).shape[1] / 6)))
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()