__author__ = 'aclapes'

import numpy as np
from os.path import isfile, exists
from os import makedirs
import cPickle
from sklearn.decomposition import PCA
from scipy.io import loadmat
from yael import yael, ynumpy

from Queue import PriorityQueue

INTERNAL_PARAMETERS = dict(
    feature_types = ['trj', 'hog', 'hof', 'mbh'],
    # dimensionality reduction
    n_samples = 1000000,  # TODO: set to 1000000
    reduction_factor = 0.5,   # keep after a fraction of the dimensions after applying pca
    # building the GMMs
    fv_gmm_k = 256,  # number of gaussian components
    fv_repr_feats = ['mu','sigma']
)

def train_reduction_maps_and_gmms(tracklets_path, videonames, st, num_videos, intermediates_path):
    if not exists(intermediates_path):
        makedirs(intermediates_path)

    total = len(videonames)
    num_samples_per_vid = int(INTERNAL_PARAMETERS['n_samples'] / float(min(st+num_videos,total)))

    # process the videos
    for i, feat_t in enumerate(INTERNAL_PARAMETERS['feature_types']):
        filepath = intermediates_path + 'fv-gmm_pca-' + feat_t + '.pkl'
        if isfile(filepath):
            print('%s -> OK' % filepath)
            continue

        D = None  # feat_t's sampled tracklets
        ptr = 0
        for j in range(st,min(st+num_videos,total)):
            # load file containing tracklets
            with open(tracklets_path + feat_t + '/' + videonames[j] + '.pkl', 'rb') as f:
                d = cPickle.load(f)
            # init sample
            if D is None:
                D = np.zeros((INTERNAL_PARAMETERS['n_samples'], d.shape[1]), dtype=np.float32)
            # create a random permutation for sampling some tracklets in this vids
            randp = np.random.permutation(d.shape[0])
            if d.shape[0] > num_samples_per_vid:
                randp = randp[:num_samples_per_vid]
            D[ptr:ptr+len(randp),:] = d[randp,:]
            ptr += len(randp)
        D = D[:ptr,:]  # cut out extra reserved space


        # (special case) trajectory features are originally positions
        if feat_t == 'trj':
            D = convert_positions_to_displacements(D)

        # scale (rootSIFT)
        D = np.sign(D) * np.sqrt(np.abs(D))

        # compute PCA map and reduce dimensionality
        pca = PCA(n_components=int(INTERNAL_PARAMETERS['reduction_factor']*D.shape[1]), copy=False)
        D = pca.fit_transform(D)

        # train GMMs for later FV computation
        D = np.ascontiguousarray(D, dtype=np.float32)
        gmm = ynumpy.gmm_learn(D, INTERNAL_PARAMETERS['fv_gmm_k'])

        with open(intermediates_path + 'fv-gmm_pca-' + feat_t + '.pkl', 'wb') as f:
            cPickle.dump(dict(pca=pca, gmm=gmm), f)


def compute_fv_representations(tracklets_path, clusters_path, intermediates_path, videonames, st, num_videos, feats_path):
    '''
    This function has been optimized, thus slightly affecting its readability.
    :param tracklets_path:
    :param intermediates_path:
    :param videonames:
    :param st:
    :param num_videos:
    :param feats_path:
    :return:
    '''
    if not exists(feats_path + 'fishervecs/'):
        if not exists(feats_path):
            makedirs(feats_path)
        makedirs(feats_path + 'fishervecs/')

    # cach'd pca and gmm
    cache = dict()
    for j, feat_t in enumerate(INTERNAL_PARAMETERS['feature_types']):
        with open(intermediates_path + 'fv-gmm_pca-' + feat_t + '.pkl', 'rb') as f:
            cache[feat_t] = cPickle.load(f)
        if not exists(feats_path + 'fishervecs/' + feat_t):
            makedirs(feats_path + 'fishervecs/' + feat_t)

    # process videos
    total = len(videonames)
    for i in range(st,min(st+num_videos,total)):
        # FV computed for all feature types? see the last in INTERNAL_PARAMETERS['feature_types']
        output_filepath = feats_path + 'fishervecs/' + INTERNAL_PARAMETERS['feature_types'][-1] + '/' + videonames[i] + '.pkl'
        if isfile(output_filepath):
            continue

        # object features used for the per-frame FV representation computation (cach'd)
        with open(tracklets_path + 'obj/' + videonames[i] + '.pkl', 'rb') as f:
            obj = cPickle.load(f)
        with open(clusters_path + videonames[i] + '.pkl', 'rb') as f:
            clusters = cPickle.load(f)

        for j, feat_t in enumerate(INTERNAL_PARAMETERS['feature_types']):
            # load video tracklets' feature
            with open(tracklets_path + feat_t + '/' + videonames[i] + '.pkl', 'rb') as f:
                d = cPickle.load(f)
                if feat_t == 'trj': # (special case)
                    d = convert_positions_to_displacements(d)

            # pre-processing
            d = np.sign(d) * np.sqrt(np.abs(d))  # scale (rootSIFT)
            d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality
            d = np.ascontiguousarray(d, dtype=np.float32)  # required in many of Yael functions

            # compute FV of the video
            # (in a global representation)
            v = ynumpy.fisher(cache[feat_t]['gmm'], d, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec

            # (in a per-frame representation)
            fids = np.unique(obj[:,0])
            V = np.zeros((len(fids),len(v)), dtype=np.float32)  # row-wise fisher vectors (matrix)
            for k, f in enumerate(fids):
                tmp = d[np.where(obj[:,0] == f)[0],:]  # hopefully this is contiguous if d already was
                V[k,:] = ynumpy.fisher(cache[feat_t]['gmm'], tmp, INTERNAL_PARAMETERS['fv_repr_feats'])  # f-th frame fisher vec

            # compute FV of the tree nodes
            T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))

            tree_global = dict()
            tree_perframe = dict()
            del T[1]  # already computed the representations (global and per-frame) of the root node
            for parent_idx, children_inds in T.iteritems():
                # (in a global representation)
                node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                tmp = d[node_inds[0]:node_inds[-1],:]
                v_node = ynumpy.fisher(cache[feat_t]['gmm'], tmp, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                tree_global[parent_idx] = v_node

                # (in a per-frame representation)
                fids = np.unique(obj[node_inds[0]:node_inds[-1],0])
                V_node = np.zeros((len(fids),len(v)), dtype=np.float32)
                for k, f in enumerate(fids):
                    tmp = d[np.where(obj[node_inds[0]:node_inds[-1],0] == f)[0],:]
                    V_node[k,:] = ynumpy.fisher(cache[feat_t]['gmm'], tmp, INTERNAL_PARAMETERS['fv_repr_feats'])
                tree_perframe[parent_idx] = V_node

            # save to disk both FV representations
            output_filepath = feats_path + 'fishervecs/' + feat_t + '/' + videonames[i] + '.pkl'
            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(v=v, V=V, tree_global=tree_global, tree_perframe=tree_perframe), f)

    print 'done'
    return




# ==============================================================================
# Helper functions
# ==============================================================================

def convert_positions_to_displacements(P):
    '''
    From positions to normalized displacements
    :param D:
    :return:
    '''

    X, Y = P[:,::2], P[:,1::2]  # X (resp. Y) are odd (resp. even) columns of D
    Vx = X[:,1:] - X[:,:-1]  # get relative displacement (velocity vector)
    Vy = Y[:,1:] - Y[:,:-1]

    D = np.zeros((P.shape[0], Vx.shape[1]+Vy.shape[1]), dtype=P.dtype)
    D[:,::2]  = Vx / np.linalg.norm(Vx, ord=2, axis=1)[:,np.newaxis]  # l2-normalize
    D[:,1::2] = Vy / np.linalg.norm(Vy, ord=2, axis=1)[:,np.newaxis]

    return D

def reconstruct_tree_from_leafs(leafs):
    """
    Given a list of leaf, recover all the nodes.

    Parameters
    ----------
    leafs:  Leafs are integers, each representing a path in the binary tree.
            For instance, a leaf value of 5 indicates the leaf is the one
            reached going throught the folliwing path: root-left-right.

    Returns
    -------
    A dictionary indicating for each node a list of all its descendents.
    Exemple:
        { 1 : [2,3,4,5,6,7,12,13,26,27],
          2 : [4,5],
          3 : [6,7,12,13,26,27],
          ...
        }
    """
    h = dict()
    q = PriorityQueue()

    # recover first intermediate nodes (direct parents from leafs)
    for path in leafs:
        parent_path = int(path/2)
        if not parent_path in h and parent_path > 1:
            q.put(-parent_path)  # deeper nodes go first (queue reversed by "-")
        h.setdefault(parent_path, []).append(path)

    # recover other intermediates notes recursevily
    while not q.empty():
        path = -q.get()
        parent_path = int(path/2)
        if not parent_path in h and parent_path > 1:  # list parent also for further processing
            q.put(-parent_path)

        h.setdefault(parent_path, [])
        h[parent_path] += ([path] + h[path])  # append children from current node to their parent

    # update with leafs
    h.update(dict((i,[i]) for i in leafs))

    return h