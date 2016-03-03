__author__ = 'aclapes'

import numpy as np
import cPickle
from os.path import join
from os.path import isfile, exists
from os import makedirs
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import StratifiedKFold
import sys
import itertools
from joblib import delayed, Parallel
from random import shuffle

import videodarwin
from tracklet_representation import normalize

def compute_ATEP_kernels(feats_path, videonames, traintest_parts, feat_types, kernels_output_path, \
                         nt=4, use_disk=False):
    """
    Compute All Tree Node Branch Evolution Pairs.
    :param feats_path:
    :param videonames:
    :param traintest_parts:
    :param feat_types:
    :param nt:
    :return:
    """

    # kernels = dict(train=dict(), test=dict())
    kernels = dict()
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]

        total = len(videonames)

        for feat_t in feat_types:
            train_filepath = join(kernels_output_path, 'train-' + feat_t + '-' + str(k) + '.pkl')
            test_filepath = join(kernels_output_path, 'test-' + feat_t + '-' + str(k) + '.pkl')
            if isfile(train_filepath) and isfile(test_filepath):
                with open(train_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_train, Kn_train = data['Kr_train'], data['Kn_train']

                with open(test_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
            else:
                kernel_repr_path = join(kernels_output_path, feat_t + '-' + str(k))
                if not exists(kernel_repr_path):
                    makedirs(kernel_repr_path)

                Parallel(n_jobs=nt, backend='threading')(delayed(construct_edge_pairs)(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl'), \
                                                                                       join(kernel_repr_path, videonames[i] + '.pkl'))
                                                         for i in xrange(total))

                try:
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                except IOError:
                    if use_disk:
                        Kr_train, Kn_train = intersection_kernel(kernel_repr_path, videonames, train_inds, nt=nt)
                    else:
                        D_train = dict()
                        for i in train_inds:
                            with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                Di = cPickle.load(f)
                            D_train.setdefault('root',[]).append(Di['root'])
                            D_train.setdefault('nodes',[]).append(Di['nodes'])

                        Kr_train, Kn_train = intersection_kernel(D_train, n_channels=2, nt=nt)

                    with open(train_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_train=Kr_train, Kn_train=Kn_train), f)

                try:
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                except IOError:
                    if use_disk:
                        Kr_test, Kn_test = intersection_kernel(kernel_repr_path, videonames, test_inds, Y=train_inds, nt=nt)
                    else:
                        if not 'D_train' in locals():
                            D_train = dict()
                            for i in train_inds:
                                with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                    Di = cPickle.load(f)
                                D_train.setdefault('root',[]).append(Di['root'])
                                D_train.setdefault('nodes',[]).append(Di['nodes'])

                        D_test = dict()
                        for i in test_inds:
                            with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                Di = cPickle.load(f)
                            D_test.setdefault('root',[]).append(Di['root'])
                            D_test.setdefault('nodes',[]).append(Di['nodes'])

                        Kr_test, Kn_test = intersection_kernel(D_test, Y=D_train, n_channels=2, nt=nt)

                    with open(test_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_test=Kr_test, Kn_test=Kn_test), f)

            # Do not use the parent
            # kernels.setdefault('train',{}).setdefault(feat_t,{})['root'] = [Kr_train[0]]
            # kernels['train'][feat_t]['nodes'] = [Kn_train[0]]
            # kernels.setdefault('test',{}).setdefault(feat_t,{})['root'] = [Kr_test[0]]
            # kernels['test'][feat_t]['nodes'] = [Kn_test[0]]

            # Use also the parent
            kernels.setdefault('train',{}).setdefault(feat_t,{})['root'] = [Kr_train[0], Kr_train[1]]
            kernels['train'][feat_t]['nodes'] = [Kn_train[0],Kn_train[1]]
            kernels.setdefault('test',{}).setdefault(feat_t,{})['root'] = [Kr_test[0], Kr_test[1]]
            kernels['test'][feat_t]['nodes'] = [Kn_test[0],Kn_test[1]]

    return kernels


def compute_ATNBEP_kernels(feats_path, videonames, traintest_parts, feat_types, kernels_output_path, \
                           nt=4, use_disk=False):
    """
    Compute All Tree Node Branch Evolution Pairs.
    :param feats_path:
    :param videonames:
    :param traintest_parts:
    :param feat_types:
    :param nt:
    :return:
    """

    kernels = dict()
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]

        total = len(videonames)

        for feat_t in feat_types:
            train_filepath = join(kernels_output_path, 'train-' + feat_t + '-' + str(k) + '.pkl')
            test_filepath = join(kernels_output_path, 'test-' + feat_t + '-' + str(k) + '.pkl')
            if isfile(train_filepath) and isfile(test_filepath):
                with open(train_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                with open(test_filepath, 'rb') as f:
                    data = cPickle.load(f)
                    Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
            else:
                kernel_repr_path = join(kernels_output_path, feat_t + '-' + str(k))
                if not exists(kernel_repr_path):
                    makedirs(kernel_repr_path)

                Parallel(n_jobs=nt, backend='threading')(delayed(construct_branch_evolutions)(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl'), \
                                                                                              join(kernel_repr_path, videonames[i] + '.pkl'))
                                                               for i in xrange(total))

                try:
                    with open(train_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_train, Kn_train = data['Kr_train'], data['Kn_train']
                except IOError:
                    if use_disk:
                        Kr_train, Kn_train = intersection_kernel(kernel_repr_path, videonames, train_inds, nt=nt)
                    else:
                        D_train = dict()
                        for i in train_inds:
                            with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                Di = cPickle.load(f)
                            D_train.setdefault('root',[]).append(Di['root'])
                            D_train.setdefault('nodes',[]).append(Di['nodes'])

                        Kr_train, Kn_train = intersection_kernel(D_train, nt=nt)
                    with open(train_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_train=Kr_train, Kn_train=Kn_train), f)

                try:
                    with open(test_filepath, 'rb') as f:
                        data = cPickle.load(f)
                        Kr_test, Kn_test = data['Kr_test'], data['Kn_test']
                except IOError:
                    if use_disk:
                        Kr_test, Kn_test = intersection_kernel(kernel_repr_path, videonames, test_inds, Y=train_inds, nt=nt)
                    else:
                        if not 'D_train' in locals():
                            D_train = dict()
                            for i in train_inds:
                                with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                    Di = cPickle.load(f)
                                D_train.setdefault('root',[]).append(Di['root'])
                                D_train.setdefault('nodes',[]).append(Di['nodes'])

                        D_test = dict()
                        for i in test_inds:
                            with open(join(kernel_repr_path, videonames[i] + '.pkl'), 'rb') as f:
                                Di = cPickle.load(f)
                            D_test.setdefault('root',[]).append(Di['root'])
                            D_test.setdefault('nodes',[]).append(Di['nodes'])

                        Kr_test, Kn_test = intersection_kernel(D_test, Y=D_train, nt=nt)

                    with open(test_filepath, 'wb') as f:
                        cPickle.dump(dict(Kr_test=Kr_test, Kn_test=Kn_test), f)

            kernels.setdefault('train',{}).setdefault(feat_t,{})['root'] = Kr_train
            kernels['train'][feat_t]['nodes'] = Kn_train
            kernels.setdefault('test',{}).setdefault(feat_t,{})['root'] = Kr_test
            kernels['test'][feat_t]['nodes'] = Kn_test

            # kernels.setdefault('train',{}).setdefault(feat_t,{})['root'] = [Kr_train[0], Kr_train[1]]
            # kernels['train'][feat_t]['nodes'] = [Kn_train[0],Kn_train[1]]
            # kernels.setdefault('test',{}).setdefault(feat_t,{})['root'] = [Kr_test[0], Kr_test[1]]
            # kernels['test'][feat_t]['nodes'] = [Kn_test[0],Kn_test[1]]

    return kernels


# ==============================================================================
# Helper functions
# ==============================================================================

def construct_edge_pairs(feat_repr_filepath, output_filepath):
    if not exists(output_filepath):
        try:
            with open(feat_repr_filepath, 'rb') as f:
                data = cPickle.load(f)

            root, nodes = _construct_edge_pairs(data)

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(root=root, nodes=nodes), f)
        except IOError:
            sys.stderr.write('# ERROR: missing training instance'
                             ' {}\n'.format(feat_repr_filepath))
            sys.stderr.flush()
            quit()

    return


def _construct_edge_pairs(data, dtype=np.float32):
    """
    A tree is a list of edges, with each edge as the concatenation of the repr. of parent and child nodes.
    :param data:
    :return root, edges:
    """
    root = (data['tree'][1].astype(dtype=dtype), np.zeros(data['tree'][1].shape, dtype=dtype))

    edges = []
    for id in data['tree'].keys():
        if id > 1:
            e = (data['tree'][id].astype('float32'), data['tree'][int(id/2.)].astype('float32'))
            edges.append(e)

    return root, edges

def construct_branch_evolutions(input_filepath, output_filepath):
    if not exists(output_filepath):
        try:
            with open(input_filepath, 'rb') as f:
                data = cPickle.load(f)

            root, nodes = _construct_branch_evolutions(data)

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(root=root, nodes=nodes), f)
        except IOError:
            sys.stderr.write('# ERROR: missing training instance'
                             ' {}\n'.format(input_filepath))
            sys.stderr.flush()
            quit()

    return

def _construct_branch_evolutions(data, dtype=np.float32):
    root = (np.array([0],dtype=dtype), np.array([0],dtype=dtype))

    branches = []
    for (id_i, x) in data['tree'].iteritems():
        if id_i > 1:
            # construct the path matrix
            X = []
            id_j = id_i
            while id_j > 0:
                X.append(data['tree'][id_j])
                id_j /= 2

            branches.append(normalize(videodarwin.darwin(np.array(X))))

    return root, branches


def intersection_kernel(input_path, videonames, X, Y=None, nt=1, verbose=True):
    points = []

    if Y is None:
        # generate combinations
        points += [(i,i) for i in xrange(len(X))]  # diagonal
        points += [p for p in itertools.combinations(np.arange(len(X)),2)]  # upper-triangle combinations
        is_symmetric = True
        Y = X
    else:
        # generate product
        points += [ p for p in itertools.product(*[np.arange(len(X)),np.arange(len(Y))]) ]
        is_symmetric = False

    if verbose:
        print('Computing fast %dx%d intersection kernel ...\n' % (len(X),len(Y)))

    step = np.int(np.floor(len(points)/nt)+1)

    shuffle(points)  # so all threads have similar workload
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(_intersection_kernel)(input_path, videonames, X, Y, points[i*step:((i+1)*step if (i+1)*step < len(points) else len(points))],
                                                                                 n_channels=n_channels, tid=i, verbose=True)
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

def intersection_kernel(X, Y=None, n_channels=1, nt=1, verbose=True):
    points = []

    X['root'], X['nodes'] = np.abs(X['root']), [np.abs(np.array(x)) for x in X['nodes']]
    if Y is None:
        # generate combinations
        points += [(i,i) for i in xrange(len(X['root']))]  # diagonal
        points += [p for p in itertools.combinations(np.arange(len(X['root'])),2)]  # upper-triangle combinations
        is_symmetric = True
        Y = X
    else:
        Y['root'], Y['nodes'] = np.abs(Y['root']), [np.abs(np.array(y)) for y in Y['nodes']]
        # generate product
        points += [ p for p in itertools.product(*[np.arange(len(X['root'])),np.arange(len(Y['root']))]) ]
        is_symmetric = False

    if verbose:
        print('Computing fast %dx%d intersection kernel ...\n' % (len(X['root']),len(Y['root'])))

    step = np.int(np.floor(len(points)/nt)+1)

    shuffle(points)  # so all threads have similar workload
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(_intersection_kernel)(X, Y, points[i*step:((i+1)*step if (i+1)*step < len(points) else len(points))],
                                                                                 n_channels=n_channels, tid=i, verbose=True)
                                                   for i in xrange(nt))

    # aggregate results of parallel computations
    Kr, Ke = ret[0][0], ret[0][1]
    for r in ret[1:]:
        Kr += r[0]
        Ke += r[1]

    # if symmetric, replicate upper to lower triangle matrix
    if is_symmetric:
        for i in xrange(n_channels):
            Kr[i] += np.triu(Kr[i],1).T
            Ke[i] += np.triu(Ke[i],1).T

    return Kr, Ke


def _intersection_kernel(input_path, videonames, X, Y, points, tid=None, verbose=True):
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
    Kn = Kr.copy()

    sorted_points = sorted(points)  # sort set of points using the i-th index
    prev_i = -1
    for pid,(i,j) in enumerate(sorted_points):
        if verbose:
            print('[Parallel intersection kernel] Thread %d, progress = %.1f%%]' % (tid,100.*(pid+1)/len(points)))
        # i-th tree already loaded, do not reload
        if prev_i < i:
            with open(join(input_path, videonames[i] + '.pkl'), 'rb') as f:
                Di = cPickle.load(f)
            Di['root'], Di['nodes'] = np.abs(Di['root']), np.abs(Di['nodes'])
            prev_i = i
        # always reload j-th tree
        with open(join(input_path, videonames[j] + '.pkl'), 'rb') as f:
            Dj = cPickle.load(f)
        Dj['root'], Dj['nodes'] = np.abs(Dj['root']), np.abs(Dj['nodes'])

        Kr[i,j] = np.minimum(Di['root'], Dj['root']).sum()

        # pair-wise intersection of edges' histograms
        sum_nodes = 0.
        for node_i in xrange(len(Di['nodes'])):
            for node_j in xrange(len(Dj['nodes'])):
                sum_nodes += np.minimum(Di['nodes'][node_i], Dj['nodes'][node_j]).sum()
        Kn[i,j] = sum_nodes / (len(Di['nodes']) * len(Dj['nodes']))

    return Kr, Kn

def _intersection_kernel(X, Y, points, n_channels=1, tid=None, verbose=True):
    """
    Compute the ATEP kernel.
    :param X:
    :param Y:
    :param points: pairs of distances to compute among (i,j)-indexed rows of X and Y respectively.
    :param tid: thread ID for verbosing purposes
    :param verbose:
    :return:
    """
    Kr = np.zeros((n_channels,len(X['root']),len(Y['root'])), dtype=np.float32)  # root kernel
    Kn = Kr.copy()

    for pid,(i,j) in enumerate(points):
        if verbose:
            print('[Parallel intersection kernel] Thread %d, progress = %.1f%%]' % (tid,100.*(pid+1)/len(points)))

        for k in xrange(n_channels):
            Kr[k,i,j] = np.minimum(X['root'][i][k], Y['root'][j][k]).sum()

        # pair-wise intersection of edges' histograms
        sum_nodes = np.zeros((n_channels), dtype=np.float64)
        for node_i in xrange(len(X['nodes'][i])):
            for node_j in xrange(len(Y['nodes'][j])):
                for k in xrange(n_channels):
                    sum_nodes[k] += np.minimum(X['nodes'][i][node_i][k], Y['nodes'][j][node_j][k]).sum()

        for k in xrange(n_channels):
            Kn[k,i,j] = sum_nodes[k] / (len(X['nodes'][i]) * len(Y['nodes'][j]))

    return Kr, Kn