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

from tracklet_representation import get_root_and_edges

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
        for feat_p in feats_path:
            for feat_t in feat_types:
                train_filepath = join(feat_p, 'ACP_train-' + feat_t + '-' + str(k) + '.pkl')
                test_filepath = join(feat_p, 'ACP_test-' + feat_t + '-' + str(k) + '.pkl')
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
                        input_filepath = join(feat_p, feat_t, videonames[i] + '-' + str(k) + '.pkl')
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
                        Kr_train, Ke_train = ACP_kernel(trees[train_inds], nt=nt)
                        with open(train_filepath, 'wb') as f:
                            cPickle.dump(dict(Kr_train=Kr_train, Ke_train=Ke_train), f)

                    try:
                        with open(test_filepath, 'rb') as f:
                            data = cPickle.load(f)
                            Kr_test, Ke_test = data['Kr_test'], data['Ke_test']
                    except IOError:
                        Kr_test, Ke_test = ACP_kernel(trees[test_inds], Y=trees[train_inds], nt=nt)
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
