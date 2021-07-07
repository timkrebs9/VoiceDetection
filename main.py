import os
from utils.hmm.gmmhmm import GMM
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt

from utils.filt.filter import *
from utils.builder.dataset import dataSetBuilder


# 1. Step: Vorverarbeitung
#   1.1 Abtastung
#   1.2 Filterung
#   1.3 Trasnformation
#   1.4 Merkmalsvektor
#   1.5 Cepstrum 

def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])

    plt.title(title)
    plt.tight_layout()


def main():
    # Get the dir of training_data
    trainData = "./train_audio/"
    testData = "./test_audio/"

    # Build trainings dataset
    trainDataSet = dataSetBuilder(trainData)
    
    # Train GMMHMM Models
    gmm = GMM(n_components = 3, n_iters = 1, tol = 1e-4, seed = 4)
    gmm.fit(trainDataSet)

    plot_contours(trainDataSet, gmm.means, gmm.covs, 'Initial clusters')

    #build test dataset
    testDataSet = dataSetBuilder(testData)
    
    cnt = 0

    #for label in testDataSet.keys():
    #    feature = testDataSet[label]
    #    scoreList = {}
    #    for model_label in hmmModel.keys():
    #        model = hmmModel[model_label]
    #        score = model.score(feature[0])
    #        scoreList[model_label] = score
    #    predict = max(scoreList, key=scoreList.get)
    #    print("Test on true label ", label, ": predict result label is ", predict)
    #    if predict == label:
    #        cnt+=1
    #print("Final recognition rate is %.2f"%(100.0*cnt/len(testDataSet.keys())), "%")

    

if __name__ == '__main__':
    main()
    
    # Read audio file in array
        #fs, sig = scipy.io.wavfile.read("OSR/"+filename)


    # Filter the Signal with Low Pass Filter
        #sig = movingAverage(x=sig, sr=fs, cutoff)

        #x = get_features(signal=sig, sample_rate=fs)
    
    
    # get voiced frames
        #energy, vad, voiced = naive_frame_energy_vad(sig, fs, threshold=-20, win_len=0.025, win_hop=0.025)
    
    
    # plot results
        #multi_plots(data=[sig, energy, vad, voiced], titles=["Input signal (voiced + silence)", "Short time energy", "Voice activity detection", "Output signal (voiced only)"], fs=fs, plot_rows=4, step=1)

    # save voiced signal
        #scipy.io.wavfile.write("rame_energy_vad"+ filename, fs,  np.array(voiced, dtype=sig.dtype))