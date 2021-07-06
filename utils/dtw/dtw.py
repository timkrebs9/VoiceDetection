from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

import os


import time 
import librosa

# Extract labels from .txt file 
with open("utils/dtw/labels.txt") as file:
    labels = np.array([l.replace("\n", '') for l in file.readlines()])

mfcc = {}
j = 1
for i in range(len(labels)):
    y, fs = librosa.load("train_audio/{}.wav".format(i))
    j +=1
    mfcc = librosa.feature.mfcc(y, fs, n_mfcc=13)
    mfcc[i] = mfcc.T

print(mfcc.shape)
