from utils.mfcc.mfcc import mfcc
import scipy.io.wavfile


def SpeechFeature(file):
    # Read the signal
    fs, sig = scipy.io.wavfile.read(file)
    mfcc_features = mfcc(sig,nwin=int(fs * 0.03), fs=fs, nceps=12)[0]
    return mfcc_features