__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from sklearn import svm
from sklearn.metrics import accuracy_score, average_precision_score
import sys
from os import makedirs
import time


INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify_using_bovwtrees(feats_path, videonames, class_labels, train_test_idx, feat_types, w, classification_path):
    if not exists(classification_path):
        makedirs(classification_path)

    # process videos
    total = len(videonames)

    kernels_train = []
    kernels_test = []
    for feat_t in feat_types:
        train_filepath = join(classification_path, 'bowtrees_ATEP_train-' + feat_t + '.pkl')
        test_filepath = join(classification_path, 'bowtrees_ATEP_test-' + feat_t + '.pkl')
        if isfile(train_filepath) and isfile(test_filepath):
            with open(train_filepath, 'rb') as f:
                data = cPickle.load(f)
                Kr_train, Ke_train = data['Kr_train'], data['Ke_train']
            with open(test_filepath, 'rb') as f:
                data = cPickle.load(f)
                Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
        else:
            try:
                with open(classification_path + 'bowtrees-' + feat_t + '.pkl', 'rb') as f:
                    bowtrees = cPickle.load(f)
            except IOError:
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

                with open(classification_path + 'bowtrees-' + feat_t + '.pkl', 'wb') as f:
                    cPickle.dump(bowtrees, f)

            bowtrees = np.array(bowtrees)

            try:
                with open(train_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_train, Ke_train = data['Kr_train'], data['Ke_train']
            except IOError:
                Kr_train, Ke_train = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_test_idx[0]])
                with open(train_filepath, 'wb') as f:
                    cPickle.dump(dict(Kr_train=Kr_train, Ke_train=Ke_train), f)

            try:
                with open(test_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
            except IOError:
                Kr_test, Ke_test = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_test_idx[0]], bowtrees_te=bowtrees[train_test_idx[1]])
                with open(test_filepath, 'wb') as f:
                    cPickle.dump(dict(Kr_test=Kr_test, Ke_test=Ke_test), f)

        K_train = w*Kr_train + (1-w)*Ke_train
        feat_norm = 1.0 / np.mean(K_train)
        kernels_train.append(feat_norm * K_train)

        K_test = w*Kr_test + (1-w)*Ke_test
        kernels_test.append(feat_norm * K_test)

    results = _train_and_classify(kernels_train, kernels_test, feat_types, class_labels, train_test_idx)
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

    for i in range(0, len(bowtrees_tr)):
        bowtrees_tr[i][0] = bowtrees_tr[i][0].astype('float')
        for edge_i in range(0, len(bowtrees_tr[i][1])):
            bowtrees_tr[i][1][edge_i] = np.concatenate(bowtrees_tr[i][1][edge_i]).astype('float')

    if bowtrees_te is not None:
        for i in range(0, len(bowtrees_te)):
            bowtrees_te[i][0] = bowtrees_te[i][0].astype('float')
            for edge_i in range(0, len(bowtrees_te[i][1])):
                bowtrees_te[i][1][edge_i] = np.concatenate(bowtrees_te[i][1][edge_i]).astype('float')
    else:
        bowtrees_te = bowtrees_tr

    # init the kernel
    Kr = np.zeros((len(bowtrees_tr),len(bowtrees_te)), dtype=np.float32)
    Ke = np.zeros((len(bowtrees_tr),len(bowtrees_te)), dtype=np.float32)

    for i in range(0,len(bowtrees_tr)):
        ptr = i if bowtrees_te is None else 0
        for j in range(ptr,len(bowtrees_te)):
            # intersection between root histograms
            Kr[i,j] = intersection( bowtrees_tr[i][0], bowtrees_te[j][0] )

            # pair-wise intersection of edges' histograms
            sum_edges = 0.0
            for edge_i in range(0,len(bowtrees_tr[i][1])):
                for edge_j in range(0,len(bowtrees_te[j][1])):
                    sum_edges += intersection( bowtrees_tr[i][1][edge_i], bowtrees_te[j][1][edge_j] )
            Ke[i,j] = (1.0/len(bowtrees_tr[i])*len(bowtrees_te[j])) * sum_edges

    if bowtrees_te is None:
        Kr = Kr + np.triu(Kr,1).T
        Ke = Ke + np.triu(Ke,1).T

    return Kr, Ke


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
    c_param = 1
    clf = svm.SVC(kernel='precomputed', C=c_param, max_iter=-1, tol=1e-3)
    clf.fit(K_tr, train_labels)

    # Predict
    test_preds = clf.predict(K_te)

    # Compute accuracy and average precision
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/len(test_labels[test_labels <= 0])
    pos_acc = float(np.sum(cmp[test_labels > 0]))/len(test_labels[test_labels > 0])
    acc = (pos_acc + neg_acc) / 2.0

    # TODO: compute ap per class
    ap = average_precision_score(test_labels, test_preds, average='weighted')

    return acc, ap


def l1normalize(x):
    x_norm = np.sum(np.abs(x))
    return x / x_norm


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

