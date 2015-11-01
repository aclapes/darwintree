__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from sklearn import svm
from sklearn.metrics import accuracy_score, average_precision_score
import sys


INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify_using_bovwtrees(feats_path, videonames, class_labels, train_test_idx, feat_types):
    # process videos
    total = len(videonames)

    kernels_tr = []
    kernels_te = []
    for feat_t in feat_types:
        bowtrees = []
        for i in xrange(total):
            input_filepath = join(feats_path, feat_t, videonames[i] + '-bovwtree.pkl')
            if not isfile(input_filepath):
                sys.stderr.write('# WARNING: missing training instance'
                                 ' {}\n'.format(input_filepath))
                sys.stderr.flush()
                continue

            with open(input_filepath) as f:
                data = cPickle.load(f)

            t = get_bovwtree(data)
            bowtrees.append(t)

        K_tr = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_test_idx[0]], w=0.5)
        K_te = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_test_idx[0]], bowtrees_te=bowtrees[train_test_idx[1]], w=0.5)

        feat_norm = 1.0 / np.mean(K_tr)
        kernels_tr.append(feat_norm * K_tr)
        kernels_te.append(feat_norm * K_te)

    results = _train_and_classify(kernels_tr, kernels_te, feat_types, class_labels, train_test_idx)
    return results



# ==============================================================================
# Helper functions
# ==============================================================================

def get_bovwtree(data):
    '''
    A tree is a list of edges, with each edge as the concatenation of the repr. of parent and child nodes.
    :param data:
    :return root, edges:
    '''
    root = data['tree_global'][1]
    del data['tree_global'][1]

    edges = [(data['tree_global'][id], data['tree_global'][int(float(id)/2)] if id > 3 else root)
             for id in data['tree_global'].keys()]

    return root, edges


def compute_ATEP_kernel(bowtrees_tr, bowtrees_te=None, w=0.5):
    if bowtrees_te is None:
        bowtrees_te = bowtrees_tr

    # init the kernel
    K = np.zeros((len(bowtrees_tr),len(bowtrees_te)), dtype=np.float32)

    for i in range(0,len(bowtrees_tr)):
        ptr = i if bowtrees_te is None else 0
        for j in range(ptr,len(bowtrees_te)):
            # intersection between root histograms
            kr = intersection( bowtrees_tr[i][0], bowtrees_te[j][0] )

            # pair-wise intersection of edges' histograms
            sum_edges = 0.0
            for edge_i in range(0,len(bowtrees_tr[i][1])):
                for edge_j in range(0,len(bowtrees_te[j][1])):
                    sum_edges += intersection( np.concatenate(bowtrees_tr[i][1][edge_i]), np.concatenate(bowtrees_te[j][1][edge_j]) )
            ke = (1.0/len(bowtrees_tr[i])*len(bowtrees_te[j])) * sum_edges

            # final score is a combination of root similary and pairwise edges' similarities
            # w = 1 would be the case of classifiying using global representation and intersection kernel
            K[i,j] = w*kr + (1-w)*ke

    if bowtrees_te is None:
        K = K + np.triu(K,1).T

    return K


def _train_and_classify(kernels_tr, kernels_te, feat_types, class_labels, train_test_idx):
    # Assign weights to channels
    if INTERNAL_PARAMETERS['weights'] is None: # if not specified a priori (when channels' specification)
        weights = [1.0/len(feat_types) for i in feat_types]

    # Perform the classification
    acc_classes = []
    ap_classes = []
    for cl in xrange(class_labels.shape[1]):
        train_labels = class_labels[train_test_idx[0], cl]
        test_labels = class_labels[train_test_idx[1], cl]
        # Weight each channel accordingly
        K_tr = weights[0] * kernels_tr[0]
        K_te = weights[0] * kernels_te[0]
        for i in range(1,len(feat_types)):
            K_tr += weights[i] * kernels_tr[i]
            K_te += weights[i] * kernels_te[i]
        # Get class results
        acc, ap = _train_and_classify_binary(K_tr, K_te, train_labels, test_labels)
        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_and_classify_binary(K_tr, K_te, train_labels, test_labels):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    c_param = 100
    clf = svm.SVC(kernel='precomputed', C=c_param, max_iter=-1, tol=1e-3)
    clf.fit(K_tr, train_labels)

    # Predict
    test_preds = clf.predict(K_te)

    # Compute accuracy and average precision
    acc = accuracy_score(test_labels, test_preds)
    ap = average_precision_score(test_labels, test_preds)

    return acc, ap


def l1normalize(x):
    x_norm = np.sum(np.abs(x))
    return x.astype('float') / x_norm


def intersection(x1, x2):
    '''
    Implements:
        h(x,x') = \sum_j min(x_j/||x||_1, x_j/||x'||_1)
    For more info, refer to:
        Eq(12) from Activity representation with motion hierarchies (A. Gaidon)
    :param h1:
    :param h2:
    :param copy:
    :return:
    '''
    xnorm1 = l1normalize(x1)
    xnorm2 = l1normalize(x2)

    min_inds = xnorm2 < xnorm1
    xnorm1[min_inds] = xnorm2[min_inds]

    return np.sum(xnorm1)

