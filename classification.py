__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import StratifiedKFold
import sys
import itertools
from joblib import delayed, Parallel
from random import shuffle

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify(kernels, class_labels, traintest_parts, a, feat_types, c=[1]):
    '''
    TODO Fill this.
    :param feats_path:
    :param class_labels:
    :param traintest_parts:
    :param a:
    :param feat_types:
    :param c:
    :return:
    '''
    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]
        kernels_train, kernels_test = kernels['train'], kernels['test']
        results[k] = train_and_classify(kernels_train, kernels_test, a, feat_types, class_labels, (train_inds, test_inds), c)

    return results



# ==============================================================================
# Helper functions
# ==============================================================================

# def print_progressbar(value, size=20, percent=True):
#     """
#     Print progress bar with value as an ASCII bar in the console.
#     :param value: progress value ranging within [0-1]
#     :param size: width of the bar
#     :param percent: print the progress as a % value, if not print in the range
#     :return:
#     """
#     bar_fill = '#'*int(np.floor(size*value))+'-'*int(np.ceil(size*(1-value)))
#     bar_expr = '\r[{:}]\t{:.3}' if not percent else '\r[{:}]\t{:.1%}'
#     print(bar_expr.format(bar_fill, value)),


def train_and_classify(kernels_tr, kernels_te, a, feat_types, class_labels, train_test_idx, c=[1], nl=2):
    '''

    :param kernels_tr:
    :param kernels_te:
    :param a: trade-off parameter controlling importance of root representation vs edges representation
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param c:
    :return:
    '''
    # Assign weights to channels
    feat_weights = INTERNAL_PARAMETERS['weights']
    if feat_weights is None: # if not specified a priori (when channels' specification)
        feat_weights = [1.0/len(kernels_tr) for i in kernels_tr]

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=42)

    S = [None] * class_labels.shape[1]  # selected (best) params
    p = [None] * class_labels.shape[1]  # performances
    C = [(a,c) for k in xrange(class_labels.shape[1])]  # candidate values for params

    Rval_ap = np.zeros((class_labels.shape[1], len(a), len(c)), dtype=np.float32)
    for k in xrange(class_labels.shape[1]):
        for l in xrange(nl):
            for i, a_i in enumerate(C[k][0]):
                # Weight each channel accordingly
                Kr_tr, _ = normalize_kernel(kernels_tr[0][0])
                Ke_tr, _ = normalize_kernel(kernels_tr[0][1])
                K_tr = feat_weights[0] * (a_i*Kr_tr + (1-a_i)*Ke_tr)
                for i in range(1,len(kernels_tr)):
                    Kr_tr, _ = normalize_kernel(kernels_tr[i][0])
                    Ke_tr, _ = normalize_kernel(kernels_tr[i][1])
                    K_tr += feat_weights[i] * (a_i*Kr_tr + (1-a_i)*Ke_tr)

                for j, c_j in enumerate(C[k][1]):
                    # print l, str(i+1) + '/' + str(len(C[k][0])), str(j+1) + '/' + str(len(C[k][1]))
                    Rval_ap[k,i,j] = 0
                    for (val_tr_inds, val_te_inds) in skf:
                        # test instances not indexed directly, but a mask is created excluding negative instances
                        val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                        val_te_msk[val_tr_inds] = False
                        negatives_msk = np.negative(np.any(class_labels[tr_inds] > 0, axis=1))
                        val_te_msk[negatives_msk] = False

                        acc_tmp, ap_tmp = _train_and_classify_binary(
                            K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                            class_labels[tr_inds,k][val_tr_inds], class_labels[tr_inds,k][val_te_msk], \
                            c_j)
                        # TODO: decide what it is
                        Rval_ap[k,i,j] += acc_tmp/skf.n_folds
                        Rval_ap[k,i,j] += (ap_tmp/skf.n_folds if acc_tmp > 0.5 else 0)

            a_bidx, c_bidx = np.unravel_index(Rval_ap[k].argmax(), Rval_ap[k].shape)  # a and c bests' indices
            S[k] = (C[k][0][a_bidx], C[k][1][c_bidx])
            p[k] = Rval_ap.max()

            a_new = np.linspace(C[k][0][a_bidx-1 if a_bidx > 0 else a_bidx], C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx], len(a))
            c_new = np.linspace(C[k][1][c_bidx-1 if c_bidx > 0 else c_bidx], C[k][1][c_bidx+1 if c_bidx < len(c)-1 else c_bidx], len(c))
            C[k] = (a_new, c_new)

    # X, Y = np.meshgrid(np.linspace(0,len(c)-1,len(c)),np.linspace(0,len(a)-1,len(a)))
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for k in xrange(class_labels.shape[1]):
    #     ax = fig.add_subplot(2,5,k+1, projection='3d')
    #     ax.plot_surface(X, Y, Rval_acc[k,:,:])
    #     ax.set_zlim([0.5, 1])
    #     ax.set_xlabel('c value')
    #     ax.set_ylabel('a value')
    #     ax.set_zlabel('acc [0-1]')
    # plt.show()

    te_msk = np.ones((len(te_inds),), dtype=np.bool)
    negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
    te_msk[negatives_msk] = False

    acc_classes = []
    ap_classes = []
    for k in xrange(class_labels.shape[1]):
        a_best = S[k][0]

        # normalize kernel (dividing by the median value of training's kernel)
        Kr_tr, mr_tr = normalize_kernel(kernels_tr[0][0])
        Ke_tr, me_tr = normalize_kernel(kernels_tr[0][1])
        Kr_te, _ = normalize_kernel(kernels_te[0][0], p=mr_tr)  # p is the normalization factor
        Ke_te, _ = normalize_kernel(kernels_te[0][1], p=me_tr)

        K_tr = feat_weights[0] * (a_best*Kr_tr + (1-a_best)*Ke_tr)
        K_te = feat_weights[0] * (a_best*Kr_te + (1-a_best)*Ke_te)

        for i in range(1,len(kernels_tr)):
            Kr_tr, mr_tr = normalize_kernel(kernels_tr[i][0])
            Ke_tr, me_tr = normalize_kernel(kernels_tr[i][1])
            Kr_te, _ = normalize_kernel(kernels_te[i][0], p=mr_tr)
            Ke_te, _ = normalize_kernel(kernels_te[i][1], p=me_tr)

            K_tr += feat_weights[i] * (a_best*Kr_tr + (1-a_best)*Ke_tr)
            K_te += feat_weights[i] * (a_best*Kr_te + (1-a_best)*Ke_te)

        c_best = S[k][1]
        acc, ap = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk], c=c_best)

        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_and_classify_binary(K_tr, K_te, train_labels, test_labels, c=1.0):
    # Train
    clf = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, max_iter=-1, tol=1e-3, verbose=False)
    clf.fit(K_tr, train_labels)

    # Predict
    test_scores = clf.decision_function(K_te)
    test_preds = clf.predict(K_te)

    # Compute accuracy and average precision
    # test_preds = test_scores > 0
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/len(test_labels[test_labels <= 0])
    pos_acc = float(np.sum(cmp[test_labels > 0]))/len(test_labels[test_labels > 0])
    acc = (pos_acc + neg_acc) / 2.0

    # TODO: decide what is it
    # ap = average_precision_score(test_labels, test_preds)
    ap = average_precision_score(test_labels, test_scores)

    return acc, ap


def normalize_kernel(K, p=None):
    if p is None:
        p = 1. / float( np.median(K[K != 0]) )
    return p*K, p

