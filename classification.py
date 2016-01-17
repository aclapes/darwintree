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

def classify_using_bovw(feats_path, videonames, class_labels, traintest_parts, feat_types, classification_path, \
                        kernel='rbf', c=[1.0], gamma=[1.0]):
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

        results[k] = _train_and_classify_with_raw_data(data_train, data_test, feat_types, class_labels, (train_inds, test_inds), \
                                                    kernel, c, gamma)
    return results

def classify_using_bovwtrees(feats_path, videonames, class_labels, traintest_parts, a, feat_types, classification_path, c=[1]):
    if not exists(classification_path):
        makedirs(classification_path)

    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]
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
                bowtrees = [None] * total
                for i in xrange(total):
                    input_filepath = join(feats_path, feat_t, videonames[i] + '-bovwtree-' + str(k) + '.pkl')
                    try:
                        with open(input_filepath) as f:
                            root, edges = get_root_and_edges(cPickle.load(f), global_repr=True, dtype=np.float32)
                            bowtrees[i] = [root, edges]
                    except IOError:
                        sys.stderr.write('# WARNING: missing training instance'
                                         ' {}\n'.format(input_filepath))
                        sys.stderr.flush()

                bowtrees = np.array(bowtrees)

                try:
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Ke_train = data['Kr_train'], data['Ke_train']
                except IOError:
                    Kr_train, Ke_train = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_inds])
                    with open(train_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_train=Kr_train, Ke_train=Ke_train), f)

                try:
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
                except IOError:
                    Kr_test, Ke_test = compute_ATEP_kernel(bowtrees_tr=bowtrees[train_inds], bowtrees_te=bowtrees[test_inds])
                    with open(test_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_test=Kr_test, Ke_test=Ke_test), f)

            kernels_train.append((Kr_train,Ke_train))
            kernels_test.append((Kr_test,Ke_test))

        results = train_and_classify_with_precomp_kernels(kernels_train, kernels_test, a, feat_types, class_labels, train_test_idx, c)
    return results



# ==============================================================================
# Helper functions
# ==============================================================================

def get_bovw(data, global_repr=True, dtype=np.float32):
    drepr = 'b' if global_repr else 'B'
    return data[drepr].astype(dtype=dtype)


def get_root_and_edges(data, global_repr=True, dtype=np.float32):
    '''
    A tree is a list of edges, with each edge as the concatenation of the repr. of parent and child nodes.
    :param data:
    :return root, edges:
    '''

    drepr = 'tree_global' if global_repr else 'tree_perframe'

    root = data[drepr][1].astype(dtype=dtype)

    edges = []
    for id in data[drepr].keys():
        if id > 1:
            e = np.concatenate([data[drepr][id], data[drepr][int(id/2.)]]).astype(dtype=dtype)
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


def compute_ATEP_kernel(bowtrees_tr, bowtrees_te=None, w=0.5):
    is_symmetric = False
    if bowtrees_te is None:
        bowtrees_te = bowtrees_tr
        is_symmetric = True

    # init the kernel
    Kr = np.zeros((len(bowtrees_tr),len(bowtrees_te)), dtype=np.float32)
    Ke = np.zeros((len(bowtrees_tr),len(bowtrees_te)), dtype=np.float32)

    for i in range(0,len(bowtrees_tr)):
        ptr = i if is_symmetric is None else 0
        for j in range(ptr,len(bowtrees_te)):
            # intersection between root histograms
            Kr[i,j] = intersection( bowtrees_tr[i][0], bowtrees_te[j][0] )

            # pair-wise intersection of edges' histograms
            sum_edges = 0.0
            for edge_i in range(0,len(bowtrees_tr[i][1])):
                for edge_j in range(0,len(bowtrees_te[j][1])):
                    sum_edges += intersection( bowtrees_tr[i][1][edge_i], bowtrees_te[j][1][edge_j] )
            Ke[i,j] = sum_edges / (len(bowtrees_tr[i][1]) * len(bowtrees_te[j][1]))

    if is_symmetric:
        Kr += np.triu(Kr,1).T
        Ke += np.triu(Ke,1).T

    return Kr.T, Ke.T


def _train_and_classify_with_raw_data(data_tr, data_te, feat_types, class_labels, train_test_idx, kernel, c, gamma):
    # Assign weights to channels
    if INTERNAL_PARAMETERS['weights'] is None: # if not specified a priori (when channels' specification)
        feat_weights = [1.0/len(feat_types) for i in feat_types]

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    lb = LabelBinarizer(neg_label=-1, pos_label=1)
    lb.fit(np.arange(class_labels.shape[1]))

    skf = StratifiedKFold(lb.inverse_transform(class_labels[tr_inds]), n_folds=4, shuffle=False, random_state=42)

    Rval_acc = np.zeros((class_labels.shape[1], len(c), len(gamma)), dtype=np.float32)
    # Rval_ap = np.copy(Rval_acc)

    # Weight each channel accordingly
    X_tr = preprocessing.normalize(data_tr[0], norm='l2', axis=1)
    X_te = preprocessing.normalize(data_te[0], norm='l2', axis=1)
    for i in range(1,len(feat_types)):
        X_tr = np.hstack((X_tr, preprocessing.normalize(data_tr[i], norm='l2', axis=1)))
        X_te = np.hstack((X_te, preprocessing.normalize(data_te[i], norm='l2', axis=1)))

    for i, c_i in enumerate(c):
        for j, gamma_j in enumerate(gamma):
            for l in xrange(class_labels.shape[1]):
                for val_tr_inds, val_te_inds in skf:
                    acc_tmp, ap_tmp = _train_and_classify_binary_with_raw_data(
                        X_tr[val_tr_inds,:], X_tr[val_te_inds,:], \
                        class_labels[tr_inds,l][val_tr_inds], class_labels[tr_inds,l][val_te_inds], \
                        kernel, c=c_i, gamma=gamma_j)
                    Rval_acc[l,i,j] += acc_tmp/skf.n_folds
                    # Rval_ap[l,i,j] += ap_tmp/skf.n_folds

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
        acc, ap = _train_and_classify_binary_with_raw_data(
                        X_tr, X_te, \
                        class_labels[tr_inds,l], class_labels[te_inds,l], \
                        kernel, c=c_best, gamma=gamma_best)
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

def train_and_classify_with_precomp_kernels(kernels_tr, kernels_te, a, feat_types, class_labels, train_test_idx, c=[1]):
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
    lb = LabelBinarizer(neg_label=-1, pos_label=1)
    lb.fit(np.arange(class_labels.shape[1]))

    skf = StratifiedKFold(lb.inverse_transform(class_labels[tr_inds]), n_folds=3, shuffle=False, random_state=42)

    R_acc = np.zeros((class_labels.shape[1], len(a), len(c)), dtype=np.float32)
    R_ap = np.copy(R_acc)
    for i, a_i in enumerate(a):
        # Weight each channel accordingly
        K_tr = feat_weights[0] * (a_i*kernels_tr[0][0]+(1-a_i)*kernels_tr[0][1])
        for i in range(1,len(feat_types)):
            K_tr += feat_weights[i] * (a_i*kernels_tr[i][0]+(1-a_i)*kernels_tr[i][1])

        for j, c_j in enumerate(c):
            print a_i, c_j
            for l in xrange(class_labels.shape[1]):
                for val_tr_inds, val_te_inds in skf:
                    acc_tmp, ap_tmp = _train_and_classify_binary_with_precomp_kernels(
                        K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_tr_inds,:][:,val_te_inds], \
                        class_labels[tr_inds,l][val_tr_inds], class_labels[tr_inds,l][val_te_inds], \
                        c_j)
                    R_acc[l,i,j] += acc_tmp/skf.n_folds
                    R_ap[l,i,j] += ap_tmp/skf.n_folds

    acc_classes = []
    ap_classes = []
    for l in xrange(class_labels.shape[1]):
        # R_ap[l][R_acc[l] <= 0.5] = 0
        a_best_idx, c_best_idx = np.unravel_index(R_acc[l].argmax(), R_acc[l].shape)
        a_best, c_best = a[a_best_idx], c[c_best_idx]

        K_te = feat_weights[0] * (a_best*kernels_te[0][0]+(1-a_best)*kernels_te[0][1])
        for i in range(1,len(feat_types)):
            K_te += feat_weights[i] * (a_best*kernels_te[i][0]+(1-a_best)*kernels_te[i][1])

        acc, ap = _train_and_classify_binary_with_precomp_kernels(
                        K_tr, K_te, \
                        class_labels[tr_inds,l], class_labels[te_inds,l], \
                        c_best)

        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)

def _train_and_classify_binary_with_raw_data(X_tr, X_te, train_labels, test_labels, kernel, c, gamma=1.0):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    clf = svm.SVC(kernel=kernel, verbose=False, gamma=gamma, class_weight='balanced', C=c, max_iter=-1, tol=1e-3)
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


def _train_and_classify_binary_with_precomp_kernels(K_tr, K_te, train_labels, test_labels, c=1.0):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    # clf = svm.SVC(kernel='precomputed', C=c_param, max_iter=-1, tol=1e-3)
    clf = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, max_iter=-1, tol=1e-3, verbose=False)
    clf.fit(K_tr, train_labels)

    # Predict
    test_scores = clf.decision_function(K_te.T)
    test_preds = clf.predict(K_te.T)

    # Compute accuracy and average precision
    # test_preds = test_scores > 0
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/len(test_labels[test_labels <= 0])
    pos_acc = float(np.sum(cmp[test_labels > 0]))/len(test_labels[test_labels > 0])
    acc = (pos_acc + neg_acc) / 2.0

    ap = average_precision_score(test_labels, test_scores)

    return acc, ap


def l1normalize(x):
    return x / np.sum(np.abs(x))

def l2normalize(x):
    return x / np.sqrt(np.dot(x,x))


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

