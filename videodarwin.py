__author__ = 'aclapes'

import numpy as np
from sklearn.svm import LinearSVR

def rootSIFT(X):
    '''
    :param X: rootSIFT operation applied to elements of X (element-wise).
    Check Fisher Vectors literature.
    :return:
    '''
    return np.multiply(np.sign(X), np.sqrt(np.abs(X)))

def normalizeL2(X):
    """
    Normalize the data using L2-norm.
    :param X: each row of X is an instance
    :return: the normalized data
    """
    X = np.matrix(X)
    return X / np.sqrt(np.sum(np.multiply(X,X), axis=1))

def linearSVR(X, c_param, norm=2):
    if norm == 2:
        XX = normalizeL2(X)

    T = X.shape[0] # temporal length
    clf = LinearSVR(C=c_param, dual=False, loss='squared_epsilon_insensitive', \
                    epsilon=0.1, tol=0.001, max_iter=-1, verbose=False)
    clf.fit(XX, np.linspace(1,T,T))

    return clf.coef_

def darwin(X, c_svm_param=1):
    '''
    Computes the videodarwin representation of a multi-variate temporal series.
    :param X: a N-by-T matrix, with N the number of features and T the time instants.
    :param c_svm_param: the C regularization parameter of the linear SVM.
    :return: the videodarwin representation
    '''
    T = X.shape[0] # temporal length
    one_to_T = np.linspace(1,T,T)
    one_to_T = one_to_T[:,np.newaxis]

    V = np.cumsum(X,axis=0) / one_to_T
    u_fow = linearSVR(rootSIFT(V), c_svm_param, 2) # videodarwin

    V = np.cumsum(np.flipud(X),axis=0) / one_to_T # reverse videodarwin
    u_rev = linearSVR(rootSIFT(V), c_svm_param, 2)

    return np.concatenate([u_fow, u_rev])