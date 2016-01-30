__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, average_precision_score, pairwise
from os import makedirs
import time
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import sys
import itertools
from joblib import delayed, Parallel
from random import shuffle

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify(feats_path, videonames, class_labels, traintest_parts, a, feat_types, c=[1], nt=4):
    '''
    TODO Fill this.
    :param feats_path:
    :param videonames:
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
        # process videos
        total = len(videonames)

        kernels_train = []
        kernels_test = []
        for feat_t in feat_types:
            train_filepath = join(feats_path, 'ATEP_train-' + feat_t + '-' + str(k) + '.pkl')
            test_filepath = join(feats_path, 'ATEP_test-' + feat_t + '-' + str(k) + '.pkl')
            if isfile(train_filepath) and isfile(test_filepath):
                with open(train_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_train, Ke_train = data['Kr_train'], data['Ke_train']
                with open(test_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
            else:
                trees = [None] * total
                for i in xrange(total):
                    input_filepath = join(feats_path, feat_t, videonames[i] + '-' + str(k) + '.pkl')
                    print input_filepath  # TODO: this is debug. get rid of this line ASAP
                    try:
                        with open(input_filepath) as f:
                            root, edges = get_root_and_edges(cPickle.load(f), dtype=np.float32)
                            trees[i] = [root, edges]
                    except IOError:
                        sys.stderr.write('# ERROR: missing training instance'
                                         ' {}\n'.format(input_filepath))
                        sys.stderr.flush()
                        quit()

                    trees = np.array(trees)

                try:
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Ke_train = data['Kr_train'], data['Ke_train']
                except IOError:
                    Kr_train, Ke_train = ATEP_kernel(trees[train_inds], nt=nt)
                    with open(train_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_train=Kr_train, Ke_train=Ke_train), f)

                try:
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
                except IOError:
                    Kr_test, Ke_test = ATEP_kernel(trees[test_inds], Y=trees[train_inds], nt=nt)
                    with open(test_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_test=Kr_test, Ke_test=Ke_test), f)

            kernels_train.append((Kr_train,Ke_train))
            kernels_test.append((Kr_test,Ke_test))

        results[k] = train_and_classify(kernels_train, kernels_test, a, feat_types, class_labels, (train_inds, test_inds), c)

    return results



# ==============================================================================
# Helper functions
# ==============================================================================

def get_root_and_edges(data, dtype=np.float32):
    '''
    A tree is a list of edges, with each edge as the concatenation of the repr. of parent and child nodes.
    :param data:
    :return root, edges:
    '''
    root = data['tree'][1].astype(dtype=dtype)

    edges = []
    for id in data['tree'].keys():
        if id > 1:
            e = np.concatenate([data['tree'][id], data['tree'][int(id/2.)]]).astype(dtype=dtype)
            edges.append(e)

    return root, edges


def compute_intersection_kernel(bovw_tr, bovw_te=None):
    is_symmetric = False
    if bovw_te is None:
        bovw_te = bovw_tr
        is_symmetric = True

    # init the kernel
    K = np.zeros((len(bovw_tr),len(bovw_te)), dtype=np.float32)

    for i in range(0,len(bovw_tr)):
        ptr = i if is_symmetric is None else 0
        for j in range(ptr,len(bovw_te)):
            # intersection between root histograms
            K[i,j] = intersection(bovw_tr[i], bovw_te[j])

    if is_symmetric:
        K += np.triu(K,1).T

    return K


def print_progressbar(value, size=20, percent=True):
    """
    Print progress bar with value as an ASCII bar in the console.
    :param value: progress value ranging within [0-1]
    :param size: width of the bar
    :param percent: print the progress as a % value, if not print in the range
    :return:
    """
    bar_fill = '#'*int(np.floor(size*value))+'-'*int(np.ceil(size*(1-value)))
    bar_expr = '\r[{:}]\t{:.3}' if not percent else '\r[{:}]\t{:.1%}'
    print(bar_expr.format(bar_fill, value)),


def ATEP_kernel(X, Y=None, nt=1, verbose=True):
    points = []

    X = [(np.abs(Xr), np.abs(Xe)) for (Xr,Xe) in X]
    if Y is None:
        # generate combinations
        points += [(i,i) for i in xrange(len(X))]  # diagonal
        points += [p for p in itertools.combinations(np.arange(len(X)),2)]  # upper-triangle combinations
        is_symmetric = True
        Y = X
    else:
        Y = [(np.abs(Yr), np.abs(Ye)) for (Yr,Ye) in Y]
        # generate product
        points += [ p for p in itertools.product(*[np.arange(len(X)),np.arange(len(Y))]) ]
        is_symmetric = False

    if verbose:
        print('Computing fast %dx%d ATEP kernel ...\n' % (len(X),len(Y)))

    step = np.int(np.floor(len(points)/nt)+1)

    shuffle(points)  # so all threads have similar workload
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(_ATEP_kernel)(X, Y, points[i*step:((i+1)*step if (i+1)*step < len(points) else len(points))], tid=i, verbose=True)
                              for i in xrange(nt))
    print 'here'
    # aggregate results of parallel computations
    Kr, Ke = ret[0][0], ret[0][1]
    for r in ret[1:]:
        Kr += r[0]
        Ke += r[1]

    # if symmetric, replicate upper to lower triangle matrix
    if is_symmetric:
        Kr += np.triu(Kr,1).T
        Ke += np.triu(Ke,1).T

    return Kr, Ke


def _ATEP_kernel(X, Y, points, tid=None, verbose=True):
    """
    Compute the ATEP kernel.
    :param X:
    :param Y:
    :param points: pairs of distances to compute among (i,j)-indexed rows of X and Y respectively.
    :param tid: thread ID for verbosing purposes
    :param verbose:
    :return:
    """
    Kr = np.zeros((len(X),len(Y)), dtype=np.float32)  # root kernel
    Ke = Kr.copy()

    for pid,(i,j) in enumerate(points):
        if verbose:
            print('[Parallel ATEP kernel] Thread %d, progress = %.1f%%]' % (tid,100.*(pid+1)/len(points)))
        # intersection between root histograms
        Kr[i,j] = intersection(X[i][0], Y[j][0])

        # pair-wise intersection of edges' histograms
        sum_edges = 0.
        for edge_i in range(0, len(X[i][1])):
            for edge_j in range(0, len(Y[j][1])):
                sum_edges += intersection(X[i][1][edge_i], Y[j][1][edge_j])
        Ke[i,j] = sum_edges / (len(X[i][1]) * len(Y[j][1]))

    return Kr, Ke


def train_and_classify(kernels_tr, kernels_te, a, feat_types, class_labels, train_test_idx, c=[1], nl=1):
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
    if INTERNAL_PARAMETERS['weights'] is None: # if not specified a priori (when channels' specification)
        feat_weights = [1.0/len(feat_types) for i in feat_types]

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
                for i in range(1,len(feat_types)):
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
                        # Rval_ap[k,i,j] += (ap_tmp/skf.n_folds if acc_tmp > 0.5 else 0)

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

        for i in range(1,len(feat_types)):
            Kr_tr, mr_tr = normalize_kernel(kernels_tr[i][0])
            Ke_tr, me_tr = normalize_kernel(kernels_tr[i][1])
            Kr_te, _ = normalize_kernel(kernels_te[i][0], p=mr_tr)
            Ke_te, _ = normalize_kernel(kernels_te[i][1], p=me_tr)

            K_tr += feat_weights[i] * (a_best*Kr_tr + (1-a_best)*Ke_tr)
            K_te += feat_weights[i] * (a_best*Kr_te + (1-a_best)*Ke_te)

        c_best = S[k][1]
        acc, ap = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk], c_best)

        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_and_classify_binary(K_tr, K_te, train_labels, test_labels, c=1.0):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    # clf = svm.SVC(kernel='precomputed', C=c_param, max_iter=-1, tol=1e-3)
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


def intersection(x, y):
    '''
    Implements:
        h(x,x') = \sum_j min(|{x_j}|^{\beta}, |{y_j}|^{\beta})
    For more info, refer to:
        Eq(12) from Activity representation with motion hierarchies (A. Gaidon)
        Section 3.3 from Classification using Intersection Kernel Support Vector Machines is Efficent (S. Maji)
    :param x: a 1-D vector
    :param y: another 1-D vector
    :return:
    '''

    return np.minimum(x,y).sum()
