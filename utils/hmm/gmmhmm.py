import numpy as np
from hmmlearn import hmm

def trainer(dataset, N=5):
    # N = states of model
    # A = Übergangsmatrix
    # pi = initiale Wahrscheinlichkeit
    # Each word get a single GMM Model
    Models = {}
    states = 4
    a = 1.0/(states-2)
    tmp_p = 1.0/(states-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                                [0, tmp_p, tmp_p, tmp_p , 0], \
                                [0, 0, tmp_p, tmp_p,tmp_p], \
                                [0, 0, 0, 0.5, 0.5], \
                                [0, 0, 0, 0, 1]],dtype=np.float)


    _A = np.array([[1/3, 1/3, 1/3, 0],
                 [0, 1/3, 1/3, 1/3],
                 [0, 0, 0.5, 0.5],
                 [0, 0, 0, 1]],dtype=np.float32)

    _pi = np.array([1, 0, 0, 0], dtype=np.float32) 

    # Each label in the dataset get a HMM Model
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states, n_mix=4, \
                           transmat_prior=_A, startprob_prior=_pi, \
                           covariance_type='diag')
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        Models[label] = model
    return Models



