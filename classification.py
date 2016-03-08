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
from utils import normalize_by_median, sum_of_arrays
from copy import deepcopy

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

# def merge(input_kernels):
#     kernels_train = input_kernels[0]['train']
#     for i,k in enumerate(input_kernels):
#         for j,feat
#         kernels_train[i]['train']


def classify(input_kernels, class_labels, traintest_parts, a, feat_types, c=[1]):
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

        kernels_train = input_kernels[k]['train']
        kernels_test  = input_kernels[k]['test']
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


# def train_and_classify(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, c=[1], nl=1):
#     '''
#
#     :param kernels_tr:
#     :param kernels_te:
#     :param a: trade-off parameter controlling importance of root representation vs edges representation
#     :param feat_types:
#     :param class_labels:
#     :param train_test_idx:
#     :param c:
#     :return:
#     '''
#
#     # Assign weights to channels
#     feat_weights = INTERNAL_PARAMETERS['weights']
#     if feat_weights is None: # if not specified a priori (when channels' specification)
#         feat_weights = {feat_t : 1.0/len(input_kernels_tr) for feat_t in input_kernels_tr.keys()}
#
#     tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
#     # lb = LabelBinarizer(neg_label=-1, pos_label=1)
#
#     class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
#     skf = StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=42)
#
#     S = [None] * class_labels.shape[1]  # selected (best) params
#     p = [None] * class_labels.shape[1]  # performances
#     C = [(a,c) for k in xrange(class_labels.shape[1])]  # candidate values for params
#
#     Rval_ap = np.zeros((class_labels.shape[1], len(a), len(c)), dtype=np.float32)
#     for k in xrange(class_labels.shape[1]):
#         for l in xrange(nl):
#             for i, a_i in enumerate(C[k][0]):
#                 kernels_tr = deepcopy(input_kernels_tr)
#                 kernels_te = deepcopy(input_kernels_te)
#                 for feat_t in kernels_tr.keys():
#                     kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1,0], norm=None)
#                     kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_i[0], 1-a_i[0]], norm=None)
#                 for feat_t in kernels_te.keys():
#                     kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1, 0], norm=None)
#                     kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_i[0], 1-a_i[0]], norm=None)
#
#                 K_tr = None
#                 # Weight each channel accordingly
#                 for feat_t in kernels_tr.keys():
#                     Kr_tr, _ = normalize_by_median(kernels_tr[feat_t]['root'])
#                     Kn_tr, _ = normalize_by_median(kernels_tr[feat_t]['nodes'])
#                     if K_tr is None:
#                         K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
#                     K_tr += feat_weights[feat_t] * (a_i[1]*Kr_tr + (1-a_i[1])*Kn_tr)
#
#                 for j, c_j in enumerate(C[k][1]):
#                     # print l, str(i+1) + '/' + str(len(C[k][0])), str(j+1) + '/' + str(len(C[k][1]))
#                     Rval_ap[k,i,j] = 0
#                     for (val_tr_inds, val_te_inds) in skf:
#                         # test instances not indexed directly, but a mask is created excluding negative instances
#                         val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
#                         val_te_msk[val_tr_inds] = False
#                         negatives_msk = np.negative(np.any(class_labels[tr_inds] > 0, axis=1))
#                         val_te_msk[negatives_msk] = False
#
#                         acc_tmp, ap_tmp = _train_and_classify_binary(
#                             K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
#                             class_labels[tr_inds,k][val_tr_inds], class_labels[tr_inds,k][val_te_msk], \
#                             c_j)
#                         # TODO: decide what it is
#                         Rval_ap[k,i,j] += acc_tmp/skf.n_folds
#                         # Rval_ap[k,i,j] += (ap_tmp/skf.n_folds if acc_tmp > 0.5 else 0)
#
#             a_bidx, c_bidx = np.unravel_index(Rval_ap[k].argmax(), Rval_ap[k].shape)  # a and c bests' indices
#             S[k] = (C[k][0][a_bidx], C[k][1][c_bidx])
#             p[k] = Rval_ap.max()
#
#             # a0_new = np.linspace(C[k][0][a_bidx-1 if a_bidx > 0 else a_bidx][0], \
#             #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][0], np.sqrt(len(a)))
#             # a1_new = np.linspace(C[k][0][a_bidx-1 if a[a_bidx] > 0 else a_bidx][1], \
#             #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][1], np.sqrt(len(a)))
#             # a_new = [c for c in itertools.product(*[a0_new,a1_new])]
#             c_new = np.linspace(C[k][1][c_bidx-1 if c_bidx > 0 else c_bidx], C[k][1][c_bidx+1 if c_bidx < len(c)-1 else c_bidx], len(c))
#
#             C[k] = (a, c_new)
#
#     # X, Y = np.meshgrid(np.linspace(0,len(c)-1,len(c)),np.linspace(0,len(a)-1,len(a)))
#     # fig = plt.figure(figsize=plt.figaspect(0.5))
#     # for k in xrange(class_labels.shape[1]):
#     #     ax = fig.add_subplot(2,5,k+1, projection='3d')
#     #     ax.plot_surface(X, Y, Rval_acc[k,:,:])
#     #     ax.set_zlim([0.5, 1])
#     #     ax.set_xlabel('c value')
#     #     ax.set_ylabel('a value')
#     #     ax.set_zlabel('acc [0-1]')
#     # plt.show()
#
#     te_msk = np.ones((len(te_inds),), dtype=np.bool)
#     negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
#     te_msk[negatives_msk] = False
#
#     acc_classes = []
#     ap_classes = []
#     for k in xrange(class_labels.shape[1]):
#         a_best = S[k][0]
#         print a_best
#
#         kernels_tr = deepcopy(input_kernels_tr)
#         kernels_te = deepcopy(input_kernels_te)
#         for feat_t in kernels_tr.keys():
#             kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1, 0], norm=None)
#             kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_best[0], 1-a_best[0]], norm=None)
#         for feat_t in kernels_te.keys():
#             kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1, 0], norm=None)
#             kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_best[0], 1-a_best[0]], norm=None)
#
#         # normalize kernel (dividing by the median value of training's kernel)
#         K_tr = K_te = None
#         for feat_t in kernels_tr.keys():
#             Kr_tr, mr_tr = normalize_by_median(kernels_tr[feat_t]['root'])
#             Kn_tr, me_tr = normalize_by_median(kernels_tr[feat_t]['nodes'])
#
#             Kr_te, _ = normalize_by_median(kernels_te[feat_t]['root'], p=mr_tr)
#             Kn_te, _ = normalize_by_median(kernels_te[feat_t]['nodes'], p=me_tr)
#
#             if K_tr is None:
#                 K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
#             K_tr += feat_weights[feat_t] * (a_best[1]*Kr_tr + (1-a_best[1])*Kn_tr)
#
#             if K_te is None:
#                 K_te = np.zeros(Kr_te.shape, dtype=np.float32)
#             K_te += feat_weights[feat_t] * (a_best[1]*Kr_te + (1-a_best[1])*Kn_te)
#
#         c_best = S[k][1]
#         acc, ap = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk], c=c_best)
#
#         acc_classes.append(acc)
#         ap_classes.append(ap)
#
#     return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def train_and_classify(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, c=[1], nl=1):
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
        feat_weights = {feat_t : 1.0/len(input_kernels_tr) for feat_t in input_kernels_tr.keys()}

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
                kernels_tr = deepcopy(input_kernels_tr)
                kernels_te = deepcopy(input_kernels_te)
                for feat_t in kernels_tr.keys():
                    kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1,0,0])
                    kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_i[0], 0, a_i[2]*(1-a_i[0])])
                for feat_t in kernels_te.keys():
                    kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1,0,0])
                    kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_i[0], 0, a_i[2]*(1-a_i[0])])

                K_tr = None
                # Weight each channel accordingly
                for feat_t in kernels_tr.keys():
                    Kr_tr, _ = normalize_by_median(kernels_tr[feat_t]['root'])
                    Kn_tr, _ = normalize_by_median(kernels_tr[feat_t]['nodes'])
                    if K_tr is None:
                        K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
                    K_tr += feat_weights[feat_t] * (a_i[1]*Kr_tr + (1-a_i[1])*Kn_tr)

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
            p[k] = Rval_ap[k].max()

            # a0_new = np.linspace(C[k][0][a_bidx-1 if a_bidx > 0 else a_bidx][0], \
            #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][0], np.sqrt(len(a)))
            # a1_new = np.linspace(C[k][0][a_bidx-1 if a[a_bidx] > 0 else a_bidx][1], \
            #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][1], np.sqrt(len(a)))
            # a_new = [c for c in itertools.product(*[a0_new,a1_new])]
            c_new = np.linspace(C[k][1][c_bidx-1 if c_bidx > 0 else c_bidx], C[k][1][c_bidx+1 if c_bidx < len(c)-1 else c_bidx], len(c))

            C[k] = (a, c_new)

    # print p, np.mean(p)

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
        print k, a_best

        kernels_tr = deepcopy(input_kernels_tr)
        kernels_te = deepcopy(input_kernels_te)
        for feat_t in kernels_tr.keys():
            kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1,0,0])
            kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_best[0], 0, a_best[2]*(1-a_best[0])])
        for feat_t in kernels_te.keys():
            kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1,0,0])
            kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_best[0], 0, a_best[2]*(1-a_best[0])])

        # normalize kernel (dividing by the median value of training's kernel)
        K_tr = K_te = None
        for feat_t in kernels_tr.keys():
            Kr_tr, mr_tr = normalize_by_median(kernels_tr[feat_t]['root'])
            Kn_tr, me_tr = normalize_by_median(kernels_tr[feat_t]['nodes'])

            Kr_te, _ = normalize_by_median(kernels_te[feat_t]['root'], p=mr_tr)
            Kn_te, _ = normalize_by_median(kernels_te[feat_t]['nodes'], p=me_tr)

            if K_tr is None:
                K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
            K_tr += feat_weights[feat_t] * (a_best[1]*Kr_tr + (1-a_best[1])*Kn_tr)

            if K_te is None:
                K_te = np.zeros(Kr_te.shape, dtype=np.float32)
            K_te += feat_weights[feat_t] * (a_best[1]*Kr_te + (1-a_best[1])*Kn_te)

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
    ap = average_precision_score(test_labels, test_preds)
    # ap = average_precision_score(test_labels, test_scores)

    return acc, ap

def print_results(results):
    '''
    Print in a given format.
    :param results: array of results which is a structure {no folds x #{acc,ap} x classes}.
    :return:
    '''
    accs = np.zeros((len(results),), dtype=np.float32)
    maps = accs.copy()
    for k in xrange(len(results)):
        accs[k] = np.mean(results[k]['acc_classes'])
        maps[k] = np.mean(results[k]['ap_classes'])

    # Print the results

    for k in xrange(len(results)):
        print("#Fold, Class_name, ACC, mAP")
        print("---------------------------")
        for i in xrange(len(results[k]['acc_classes'])):
            print("%d, %s, %.1f%%, %.1f%%" % (k+1, i, results[k]['acc_classes'][i]*100, results[k]['ap_classes'][i]*100))
        print("%d, ALL classes, %.1f%%, %.1f%%" % (k+1, accs[k]*100, maps[k]*100))
        print

    print("TOTAL, All folds, ACC: %.1f%%, mAP: %.1f%%" % (np.mean(accs)*100, np.mean(maps)*100))