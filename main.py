import os
import scipy
import scipy.io.wavfile
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from utils.filt.filter import *
from utils.fft.fft import naive_frame_energy_vad
from utils.math.statistica import Statistics
from helper import *

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

    # fs = 8kHz, Mono
    filename = 'Hello2.wav'
    
    #R ead audio file in array
    fs, sig = scipy.io.wavfile.read("OSR/"+filename)

    


    # Filter the Signal with Low Pass Filter
        #sig = movingAverage(x=sig, sr=fs, cutoff)

    #x = get_features(signal=sig, sample_rate=fs)
    
    
    # get voiced frames
        #energy, vad, voiced = naive_frame_energy_vad(sig, fs, threshold=-20, win_len=0.025, win_hop=0.025)
    
    
    # plot results
        #multi_plots(data=[sig, energy, vad, voiced], titles=["Input signal (voiced + silence)", "Short time energy", "Voice activity detection", "Output signal (voiced only)"], fs=fs, plot_rows=4, step=1)

    # save voiced signal
        #scipy.io.wavfile.write("rame_energy_vad"+ filename, fs,  np.array(voiced, dtype=sig.dtype))