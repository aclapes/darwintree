
__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, average_precision_score, pairwise
import sys
from os import makedirs
import time
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify(feats_path, videonames, class_labels, traintest_parts, feat_types, classification_path, kernel='rbf', c=[1.0], gamma=[1.0]):
    if not exists(classification_path):
        makedirs(classification_path)

    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]
        # process videos
        total = len(videonames)

        data_train = []
        data_test = []
        for feat_t in feat_types:
            train_filepath = join(classification_path, 'bovw_train-' + feat_t + '-' + str(k) + '.pkl')
            test_filepath = join(classification_path, 'bovw_test-' + feat_t + '-' + str(k) + '.pkl')
            if isfile(train_filepath) and isfile(test_filepath):
                with open(train_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    D_train = data['D_train']
                with open(test_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    D_test = data['D_test']
            else:
                try:
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        D_train = data['D_train']
                except IOError:
                    bovw = [None] * len(train_inds)
                    for i, idx in enumerate(train_inds):
                        input_filepath = join(feats_path, feat_t, videonames[idx] + '-bovw-' + str(k) + '.pkl')
                        try:
                            with open(input_filepath) as f:
                                bovw[i] = get_bovw(cPickle.load(f), global_repr=True, dtype=np.float32)
                        except IOError:
                            sys.stderr.write('# WARNING: missing training instance'
                                             ' {}\n'.format(input_filepath))
                            sys.stderr.flush()

                    D_train = np.array(bovw)
                    with open(train_filepath, 'wb') as f:
                        cPickle.dump(dict(D_train=D_train), f)

                try:
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        D_test = data['D_test']
                except IOError:
                    bovw = [None] * len(test_inds)
                    for i, idx in enumerate(test_inds):
                        input_filepath = join(feats_path, feat_t, videonames[idx] + '-bovw-' + str(k) + '.pkl')
                        try:
                            with open(input_filepath) as f:
                                bovw[i] = get_bovw(cPickle.load(f), global_repr=True, dtype=np.float32)
                        except IOError:
                            sys.stderr.write('# WARNING: missing training instance'
                                             ' {}\n'.format(input_filepath))
                            sys.stderr.flush()

                    D_test = np.array(bovw)
                    with open(test_filepath, 'wb') as f:
                        cPickle.dump(dict(D_test=D_test), f)

            data_train.append(D_train)
            data_test.append(D_test)

        results[k] = _train_and_classify(data_train, data_test, feat_types, class_labels, (train_inds, test_inds), kernel, c, gamma)

    return results


# ==============================================================================
# Helper functions
# ==============================================================================

def get_bovw(data, global_repr=True, dtype=np.float32):
    drepr = 'b' if global_repr else 'B'
    return data[drepr].astype(dtype=dtype)


def _train_and_classify(data_tr, data_te, feat_types, class_labels, train_test_idx, kernel, c, gamma):
    # Assign weights to channels
    if INTERNAL_PARAMETERS['weights'] is None: # if not specified a priori (when channels' specification)
        feat_weights = [1.0/len(feat_types) for i in feat_types]

    # Weight each channel accordingly
    X_tr = preprocessing.normalize(data_tr[0], norm='l2', axis=1)
    X_te = preprocessing.normalize(data_te[0], norm='l2', axis=1)
    for i in range(1,len(feat_types)):
        X_tr = np.hstack((X_tr, preprocessing.normalize(data_tr[i], norm='l2', axis=1)))
        X_te = np.hstack((X_te, preprocessing.normalize(data_te[i], norm='l2', axis=1)))

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    lb = LabelBinarizer(neg_label=-1, pos_label=1)
    lb.fit(np.arange(class_labels.shape[1]))

    skf = StratifiedKFold(lb.inverse_transform(class_labels[tr_inds]), n_folds=4, shuffle=False, random_state=42)

    Rval_acc = np.zeros((class_labels.shape[1], len(c), len(gamma)), dtype=np.float32)
    for i, c_i in enumerate(c):
        for j, gamma_j in enumerate(gamma):
            for l in xrange(class_labels.shape[1]):
                for val_tr_inds, val_te_inds in skf:
                    acc_tmp, _ = _train_and_classify_binary(X_tr[val_tr_inds,:], X_tr[val_te_inds,:], \
                                                            class_labels[tr_inds,l][val_tr_inds], class_labels[tr_inds,l][val_te_inds], \
                                                            kernel, c=c_i, gamma=gamma_j)
                    Rval_acc[l,i,j] += acc_tmp/skf.n_folds

    # X, Y = np.meshgrid(np.linspace(0,len(gamma)-1,len(gamma)), np.linspace(0,len(c)-1,len(c)))
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for l in xrange(class_labels.shape[1]):
    #     print np.max(Rval_acc[l,:,:])
    #     ax = fig.add_subplot(2,5,l+1, projection='3d')
    #     ax.plot_surface(X, Y, Rval_acc[l,:,:])
    #     ax.set_zlim([0.5, 1])
    #     ax.set_xlabel('gamma value')
    #     ax.set_ylabel('c value')
    #     ax.set_zlabel('acc [0-1]')
    # plt.show()

    # Rte_acc = np.zeros((class_labels.shape[1], len(c), len(gamma)), dtype=np.float32)

    acc_classes = []
    ap_classes = []
    # for i, c_i in enumerate(c):
    #     for j, gamma_j in enumerate(gamma):
    for l in xrange(class_labels.shape[1]):
        best_idx = np.unravel_index(Rval_acc[l].argmax(), Rval_acc[l].shape)
        c_best, gamma_best = c[best_idx[0]], gamma[best_idx[1]]
        acc, ap = _train_and_classify_binary(X_tr, X_te, class_labels[tr_inds,l], class_labels[te_inds,l], \
                                             kernel=kernel, c=c_best, gamma=gamma_best)
        # Rte_acc[l,i,j] = acc
        acc_classes.append(acc)
        ap_classes.append(ap)

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for l in xrange(class_labels.shape[1]):
    #     print np.max(Rte_acc[l,:,:])
    #     ax = fig.add_subplot(2,5,l+1, projection='3d')
    #     ax.plot_surface(X, Y, Rte_acc[l,:,:])
    #     ax.set_zlim([0.5, 1])
    #     ax.set_xlabel('gamma value')
    #     ax.set_ylabel('c value')
    #     ax.set_zlabel('acc [0-1]')
    # plt.show()

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_and_classify_binary(X_tr, X_te, train_labels, test_labels, kernel='linear', c=1.0, gamma=1.0):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    clf = svm.SVC(kernel=kernel, verbose=False, C=c, gamma=gamma, class_weight='balanced', max_iter=-1, tol=1e-3)
    clf.fit(X_tr, train_labels)

    # Predict
    test_scores = clf.decision_function(X_te)

    # Compute accuracy and average precision
    test_preds = np.ones(test_scores.shape, dtype=np.int32)
    test_preds[test_scores < 0] = -1
    # test_preds = test_scores > 0
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/sum(test_labels <= 0)
    pos_acc = float(np.sum(cmp[test_labels > 0]))/sum(test_labels > 0)
    acc = (pos_acc + neg_acc) / 2.0

    ap = average_precision_score(test_labels, test_scores)

    return acc, ap