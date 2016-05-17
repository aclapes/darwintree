__author__ = 'aclapes'

import numpy as np
from sklearn.svm import LinearSVR
from sklearn.preprocessing import normalize

from utils import posneg, rootSIFT


def linearSVR(X, c_param, norm='l2'):
    XX = normalize(X, norm=norm, axis=1)

    T = X.shape[0] # temporal length
    clf = LinearSVR(C=c_param, dual=False, loss='squared_epsilon_insensitive', \
                    epsilon=0.1, tol=1e-3, verbose=False)  # epsilon is "-p" in C's liblinear and tol is "-e"

    clf.fit(XX, np.linspace(1,T,T))

    return clf.coef_

def darwin(X, c_svm_param=1, axis=0):
    w_fw, w_rv = _darwin(X, c_svm_param=c_svm_param, axis=axis)

    return np.concatenate([w_fw, w_rv])

def _darwin(X, c_svm_param=1, axis=0):
    '''
    Computes the videodarwin representation of a multi-variate temporal series.
    :param X: a N-by-T matrix, with N the number of features and T the time instants.
    :param c_svm_param: the C regularization parameter of the linear SVM.
    :return: the videodarwin representation
    '''

    if axis == 1:
        X = X.T

    T = X.shape[0] # temporal length
    one_to_T = np.linspace(1,T,T)
    one_to_T = one_to_T[:,np.newaxis]

    V = np.cumsum(X,axis=0) / one_to_T
    w_fw = linearSVR(get_non_linearity(V,axis=0), c_svm_param, norm='l2') # non linearity on rows (axis = 0)

    V = np.cumsum(np.flipud(X),axis=0) / one_to_T # reverse videodarwin
    w_rv = linearSVR(get_non_linearity(V,axis=0), c_svm_param, norm='l2')

    return w_fw, w_rv


def get_non_linearity(X, axis=0, copy=False):
    """
    Map the data to a non linear feature space.
    :param X:
    :param axis:
    :param copy:
    :return:
    """
    return np.sqrt(posneg(X, axis=axis, copy=copy))

# X = np.array([[10,-10,0],
#               [8, -4, 1],
#               [6,  0, 2],
#               [4,  4, 3],
#               [2,  4.5, 4],
#               [-5, 10, 5]])
#
# fw, rv =  _darwin(X)
# print fw
# print rv