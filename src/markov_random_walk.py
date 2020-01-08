# Adapted from Partially labeled classification with Markov random walks from Martin Szummer and Tommi Jaakkola
# Inspired by Muhammet Balcilar on https://github.com/balcilar/SemiSupervisedMarkovRandomWalk/blob/master/sslMarkovRandomWalks.py

import numpy as np
from scipy.spatial.distance import cdist, euclidean
import gin

@gin.configurable
class MRW(object):
    """Markov random walk clustering class"""

    def __init__(self, k=5, sigma=0.6, precision=1e-9, t=3):
        super(MRW, self).__init__()
        self.k = k
        self.sigma = sigma
        self.precision = precision
        self.t = t

        self.test_data_in = False
        self.test_data_out = False
        self.training_data_in = False
        self.training_data_out = False

    def give_test_data(self, test_data_in, test_data_out):
        #Check for existing data, add accordingly
        if self.test_data_in:
            self.test_data_in 	= np.concatenate((self.test_data_in, test_data_in))
        else:
            self.test_data_in 	= test_data_in

        #Check for existing data, add accordingly
        if self.test_data_out:
            self.test_data_out 	= np.concatenate((self.test_data_out, test_data_out))
        else:
            self.test_data_out 	= test_data_out


    def give_training_data(self, training_data_in, training_data_out):
        #Check that data is the same length
        if len(training_data_in) != len(training_data_out):
            print("Data in and Data out have different lengths.")

        #Check for existing data, add accordingly
        if self.training_data_in:
            self.training_data_in 	= np.concatenate((self.training_data_in, training_data_in))
        else:
            self.training_data_in 	= training_data_in

        #Check for existing data, add accordingly
        if self.training_data_out:
            self.training_data_out 	= np.concatenate((self.training_data_out, training_data_out))
        else:
            self.training_data_out 	= training_data_out

    def calculate_P(self):
        """Computes the P matrix usefull for the classification"""
        # Computing the distance matrix
        all_data_in = np.vstack((self.training_data_in, self.test_data_in))
        size = all_data_in.shape[0]
        d = cdist(all_data_in, all_data_in, metric='euclidean')

        # Keeping only the k-nn for each data point
        indexes = d.argsort()
        indexes = indexes[:,1:self.k] # excluding the first one since it is itself

        # Creating the weight matrix as defined in the paper
        W = np.identity(size)
        for i in range(size):
            for idx in indexes[i]:
                W[i, idx] = np.exp(-d[i, idx]/self.sigma)

        # Creating the A matrix
        A = W/W.sum(axis=0)
        At = np.linalg.matrix_power(A, self.t)

        self.P = np.transpose(At) # the proba of being in k coming from i is the same as the one from going to k from i

    def indexes(self):
        posindexes = np.where(self.training_data_out==1)[0]
        negindexes = np.where(self.training_data_out==-1)[0]
        return posindexes, negindexes

    def prior(self, posindexes, negindexes):
        # +1 points are assigned a proba 1 and -1 a proba 0
        # Unlabeled points are assigned a proba 0.5
        size = self.training_data_in.shape[0]+self.test_data_in.shape[0]
        P0 = 0.5*np.ones((size, 1))
        P0[posindexes] = 1.
        P0[negindexes] = 0.
        return P0

    def compute_likelihood(self, p_0, posindexes, negindexes):
        """The computation of the posterior is made by using the EM esimation"""
        L = self.training_data_in.shape[0]

        # E-step
        Cold = p_0
        C = self.P * Cold
        # M-step
        C = C[:,posindexes].sum(axis=1)/C[:,:L].sum(axis=1)
        # Re-assign the known values
        C[posindexes] = 1.
        C[negindexes] = 0.
        C[np.isnan(C)] = 0.5

        K=0
        while euclidean(C, Cold) > self.precision:
            K=K+1
            print("JE PRINT K", K)
            Cold = C
            C = self.P * Cold[:,np.newaxis]
            # M-step
            C = C[:,posindexes].sum(axis=1)/C[:,:L].sum(axis=1)
            # Re-assign the known values
            C[posindexes] = 1.
            C[negindexes] = 0.
            C[np.isnan(C)] = 0.5

        return C

    def classify_dataset(self):
        self.calculate_P()
        posindexes, negindexes = self.indexes()
        p_0 = self.prior(posindexes, negindexes)
        results = self.compute_likelihood(p_0, posindexes, negindexes)

        classifications = np.ones(results.shape[0])
        classifications[classifications<0.5]=-1

        classifications = classifications[self.training_data_in.shape[0]:]
        misclassification = np.sum(classifications == self.test_data_out)/len(classifications)

        return missclassification
