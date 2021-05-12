import numpy as np
import math 
import logging


from scipy.fftpack import dct


def preemphasis(signal, alpha=0.97):
    """The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies. A pre-emphasis filter 
    is useful in several ways: 
    (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies, 
    (2) avoid numerical problems during the Fourier transform operation and 
    (3) may also improve the Signal-to-Noise Ratio (SNR).
    The pre-emphasis filter can be applied to a signal x using the first order filter in the following equation:
                                        y(t)=x(t)−αx(t−1)
    which can be easily implemented using the following line, where typical values for the filter coefficient (α) are 0.95 or 0.97

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.

    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def framesig(sig, frame_len, frame_step):
    """Frame a signal into overlapping frames.


    :param sig: the audio signal to frame. (emphasized signal!!!!)
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    sig_len = len(sig)
    frame_len = int(round(frame_len)+1)     # because normal round(1.5) = 1
    frame_step = int(round(frame_step)+1)
    if sig_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * sig_len - frame_len) / frame_step))

    pad_len = int((num_frames - 1) * frame_step + frame_len)
    zeros = np.zeros((pad_len - sig_len,))
    padsignal = num_frames * frame_step + frame_len
    z = np.zeros((padsignal - sig_len))
    pad_signal = np.append(sig, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    # After slicing the signal into frames, we apply a window function such as 
    # the Hamming window to each frame. A Hamming window has the following form:
    #                   w[n]=0.54−0.46cos(2πn/N−1)
    # where, 0 ≤ n ≤ N−1, N is the window length
    #     
    frames *= np.hamming(frame_len)  # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    return frames


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.', np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01, nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97, winfunc=lambda x:np.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = preemphasis(signal,preemph)
    frames = framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.) * np.sin(np.pi * n/L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13, nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True, winfunc=lambda x:np.ones((x,))):
    nfft = calculate_nfft(samplerate=samplerate, winlen=winlen)

    features,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    features = np.log(features)
    features = dct(features, type=2, axis=1, norm='ortho')[:,:numcep]
    features = lifter(features,ceplifter)
    if appendEnergy: features[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    
    return features


def delta(mfcc_features, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(mfcc_features)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(mfcc_features)
    padded = np.pad(mfcc_features, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
#
#def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01, nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97, winfunc=lambda x:np.ones((x,))):
#    """Compute log Mel-filterbank energy features from an audio signal.
#    :param signal: the audio signal from which to compute features. Should be an N*1 array
#    :param samplerate: the sample rate of the signal we are working with, in Hz.
#    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
#    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
#    :param nfilt: the number of filters in the filterbank, default 26.
#    :param nfft: the FFT size. Default is 512.
#    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
#    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
#    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
#    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
#    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
#    """
#    features,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
#    return np.log(features)

def get_features(signal, sample_rate, num_delta=5):
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=12)
    wav_features = np.empty(shape=[mfcc_features.shape[0], 0])
    
    delta_features = delta(mfcc_features, num_delta)
    wav_features = np.append(wav_features, delta_features, 1)
    wav_features = np.append(mfcc_features, wav_features, 1)

    return wav_features