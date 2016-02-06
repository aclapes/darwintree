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

import videodarwin


def compute_ATEP(feats_path, videonames, traintest_parts, feat_types, nt=4):
    """
    Compute All Tree Edge Pairs.
    :param feats_path:
    :param videonames:
    :param traintest_parts:
    :param feat_types:
    :param nt:
    :return:
    """
    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]
        # process videos
        total = len(videonames)

        kernels = dict()
        for feat_p in (feats_path if type(feats_path) is list else [feats_path]):
            for feat_t in feat_types:
                train_filepath = join(feat_p, 'ATEP_train-' + feat_t + '-' + str(k) + '.pkl')
                test_filepath = join(feat_p, 'ATEP_test-' + feat_t + '-' + str(k) + '.pkl')
                if isfile(train_filepath) and isfile(test_filepath):
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                else:
                    trees = [None] * total
                    for i in xrange(total):
                        input_filepath = join(feat_p, feat_t, videonames[i] + '-' + str(k) + '.pkl')
                        try:
                            with open(input_filepath) as f:
                                data = cPickle.load(f)
                                root, nodes = construct_edge_pairs(data, dtype=np.float32)
                                trees[i] = [root, nodes]
                        except IOError:
                            sys.stderr.write('# ERROR: missing training instance'
                                             ' {}\n'.format(input_filepath))
                            sys.stderr.flush()
                            quit()

                        trees = np.array(trees)

                    try:
                        with open(train_filepath, 'rb') as f:
                            data = cPickle.load(f)
                            Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                    except IOError:
                        Kr_train, Kn_train = intersection_kernel(trees[train_inds], nt=nt)
                        with open(train_filepath, 'wb') as f:
                            cPickle.dump(dict(Kr_train=Kr_train, Kn_train=Kn_train), f)

                    try:
                        with open(test_filepath, 'rb') as f:
                            data = cPickle.load(f)
                            Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                    except IOError:
                        Kr_test, Kn_test = intersection_kernel(trees[test_inds], Y=trees[train_inds], nt=nt)
                        with open(test_filepath, 'wb') as f:
                            cPickle.dump(dict(Kr_test=Kr_test, Kn_test=Kn_test), f)

                kernels.setdefault('train',[]).append((Kr_train,Kn_train))
                kernels.setdefault('test', []).append((Kr_test,Kn_test))

    return kernels

def compute_ATNBEP(feats_path, videonames, traintest_parts, feat_types, nt=4):
    """
    Compute All Tree Node Branch Evolution Pairs.
    :param feats_path:
    :param videonames:
    :param traintest_parts:
    :param feat_types:
    :param nt:
    :return:
    """
    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]
        # process videos
        total = len(videonames)

        kernels = dict()
        for feat_p in (feats_path if type(feats_path) is list else [feats_path]):
            for feat_t in feat_types:
                train_filepath = join(feat_p, 'ATNBEP_train-' + feat_t + '-' + str(k) + '.pkl')
                test_filepath = join(feat_p, 'ATNBEP_test-' + feat_t + '-' + str(k) + '.pkl')
                if isfile(train_filepath) and isfile(test_filepath):
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                else:
                    trees = [None] * total
                    for i in xrange(total):
                        input_filepath = join(feat_p, feat_t, videonames[i] + '-' + str(k) + '.pkl')
                        try:
                            with open(input_filepath) as f:
                                data = cPickle.load(f)
                                root, nodes = construct_branch_evolutions(data, dtype=np.float32)
                                trees[i] = [root, nodes]
                        except IOError:
                            sys.stderr.write('# ERROR: missing training instance'
                                             ' {}\n'.format(input_filepath))
                            sys.stderr.flush()
                            quit()

                        trees = np.array(trees)

                    try:
                        with open(train_filepath, 'rb') as f:
                            data = cPickle.load(f)
                            Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                    except IOError:
                        Kr_train, Kn_train = intersection_kernel(trees[train_inds], nt=nt)
                        with open(train_filepath, 'wb') as f:
                            cPickle.dump(dict(Kr_train=Kr_train, Kn_train=Kn_train), f)

                    try:
                        with open(test_filepath, 'rb') as f:
                            data = cPickle.load(f)
                            Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                    except IOError:
                        Kr_test, Kn_test = intersection_kernel(trees[test_inds], Y=trees[train_inds], nt=nt)
                        with open(test_filepath, 'wb') as f:
                            cPickle.dump(dict(Kr_test=Kr_test, Kn_test=Kn_test), f)

                kernels.setdefault('train',[]).append((Kr_train,Kn_train))
                kernels.setdefault('test', []).append((Kr_test,Kn_test))

    return kernels


# ==============================================================================
# Helper functions
# ==============================================================================

def construct_edge_pairs(data, dtype=np.float32):
    """
    A tree is a list of edges, with each edge as the concatenation of the repr. of parent and child nodes.
    :param data:
    :return root, edges:
    """
    root = data['tree'][1].astype(dtype=dtype)

    edges = []
    for id in data['tree'].keys():
        if id > 1:
            e = np.concatenate( [data['tree'][id], data['tree'][int(id/2.)]] ).astype(dtype=dtype)
            edges.append(e)

    return root, edges


def construct_branch_evolutions(data, dtype=np.float32):
    root = data['tree'][1].astype(dtype=np.float32)  # copy root

    branches = []
    for (id_i, x) in data['tree'].iteritems():
        if id_i > 1:
            # construct the path matrix
            X = []
            id_j = id_i
            while id_j > 0:
                X.append(data['tree'][id_j])
                id_j /= 2

            # compute evolution of the branch
            w = videodarwin.darwin(np.array(X, dtype=np.float32))

            # build the node representation its representation itself + the videodarwin of the path
            b = np.concatenate([x,w]).astype(dtype=dtype)
            branches.append(b)

    return root, branches


def intersection_kernel(X, Y=None, nt=1, verbose=True):
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
        print('Computing fast %dx%d intersection kernel ...\n' % (len(X),len(Y)))

    step = np.int(np.floor(len(points)/nt)+1)

    shuffle(points)  # so all threads have similar workload
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(_ATEP_kernel)(X, Y, points[i*step:((i+1)*step if (i+1)*step < len(points) else len(points))], tid=i, verbose=True)
                              for i in xrange(nt))

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


def _intersection_kernel(X, Y, points, tid=None, verbose=True):
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
            print('[Parallel intersection kernel] Thread %d, progress = %.1f%%]' % (tid,100.*(pid+1)/len(points)))
        # intersection between root histograms
        Kr[i,j] = np.minimum(X[i][0], Y[j][0]).sum()

        # pair-wise intersection of edges' histograms
        sum_edges = 0.
        for edge_i in range(0, len(X[i][1])):
            for edge_j in range(0, len(Y[j][1])):
                sum_edges += np.minimum(X[i][1][edge_i], Y[j][1][edge_j]).sum()
        Ke[i,j] = sum_edges / (len(X[i][1]) * len(Y[j][1]))

    return Kr, Ke