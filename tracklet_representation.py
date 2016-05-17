__author__ = 'aclapes'

import numpy as np
from os.path import join
from os.path import isfile, exists
from os import makedirs
import cPickle
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
from yael import ynumpy
import time
import sys
from joblib import delayed, Parallel
import videodarwin


from Queue import PriorityQueue


INTERNAL_PARAMETERS = dict(
    # dimensionality reduction
    n_samples = 1000000, #1000*256,  # See paper of "A robust and efficient video representation for action recognition"
    reduction_factor = 0.5,   # keep after a fraction of the dimensions after applying pca
    # bulding codebooks
    bovw_codebook_k = 4000,
    bovw_lnorm = 1,
    # building GMMs
    fv_gmm_k = 256,  # number of gaussian components
    fv_repr_feats = ['mu','sigma']
)

def compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                         nt=4, pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_bovw_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                                [i], feat_types, feats_path, \
                                                                                pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path, verbose=verbose)
                                                           for i in xrange(len(videonames)))

def compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                       nt=4, pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_fv_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                              [i], feat_types, feats_path, \
                                                                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path, verbose=verbose)
                                                           for i in xrange(len(videonames)))

def compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                       nt=4, pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_vd_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                              [i], feat_types, feats_path, \
                                                                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path, verbose=verbose)
                                                           for i in xrange(len(videonames)))


# ==============================================================================
# Main functions
# ==============================================================================


def _compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                              pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        # cach'd pca and gmm
        for j, feat_t in enumerate(feat_types):
            try:
                makedirs( join(feats_path, feat_t + '-' + str(k)) )
            except OSError:
                pass

        cache = None

        # process videos
        total = len(videonames)
        for i in indices:
            # FV computed for all feature types? see the last in INTERNAL_PARAMETERS['feature_types']
            all_done = np.all([isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl'))
                               for feat_t in feat_types])
            if all_done:
                if verbose:
                    print('[_compute_bovw_descriptors] %s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(join(intermediates_path, 'bovw' + ('-' if pca_reduction else '-nopca-') + feat_t + '-' + str(k) + '.pkl'), 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(join(tracklets_path, 'obj', videonames[i] + '.pkl'), 'rb') as f:
                obj = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                # load video tracklets' feature
                with open(join(tracklets_path, feat_t, videonames[i] + '.pkl'), 'rb') as f:
                    d = cPickle.load(f)

                if feat_t == 'trj': # (special case)
                    d = convert_positions_to_displacements(d)

                if feat_t == 'mbh':
                    dx = preprocessing.normalize(d[:,:d.shape[1]/2], norm='l1', axis=1)
                    dy = preprocessing.normalize(d[:,d.shape[1]/2:], norm='l1', axis=1)
                    d = np.hstack((dx,dy))
                else:
                    d = preprocessing.normalize(d, norm='l1', axis=1)

                d = rootSIFT(d)

                if pca_reduction:
                    d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality

                output_filepath = join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')
                # compute BOVW of the video
                if not treelike:
                    b = bovw(cache[feat_t]['codebook'], d)
                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=b), f)

                else:  # or separately the BOVWs of the tree nodes
                    with open(join(clusters_path, videonames[i] + '.pkl'), 'rb') as f:
                        clusters = cPickle.load(f)

                    bovwtree = dict()
                    if len(clusters['tree']) == 1:
                        bovwtree[1] = bovw(cache[feat_t]['codebook'], d)
                    else:
                        T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                        for parent_idx, children_inds in T.iteritems():
                            # (in a global representation)
                            node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                            bovwtree[parent_idx] = bovw(cache[feat_t]['codebook'], d[node_inds,:])  # bovw vec

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=bovwtree), f)

            elapsed_time = time.time() - start_time
            if verbose:
                print('[_compute_bovw_descriptors] %s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def _compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                            pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(join(feats_path, feat_t + '-' + str(k)))
            except OSError:
                pass

        cache = None

        # process videos
        total = len(videonames)
        for i in indices:
            # FV computed for all feature types? see the last in INTERNAL_PARAMETERS['feature_types']
            all_done = np.all([isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl'))
                               for feat_t in feat_types])
            if all_done:
                if verbose:
                    print('[_compute_fv_descriptors] %s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(join(intermediates_path, 'gmm' + ('-' if pca_reduction else '-nopca-') + feat_t + '-' + str(k) + '.pkl'), 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(join(tracklets_path, 'obj', videonames[i] + '.pkl'), 'rb') as f:
                obj = cPickle.load(f)
            with open(join(clusters_path, videonames[i] + '.pkl'), 'rb') as f:
                clusters = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                if isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')):
                    continue

                # load video tracklets' feature
                with open(join(tracklets_path, feat_t, videonames[i] + '.pkl'), 'rb') as f:
                    d = cPickle.load(f)

                if feat_t == 'trj': # (special case)
                    d = convert_positions_to_displacements(d)

                if feat_t == 'mbh':
                    dx = preprocessing.normalize(d[:,:d.shape[1]/2], norm='l1', axis=1)
                    dy = preprocessing.normalize(d[:,d.shape[1]/2:], norm='l1', axis=1)
                    d = np.hstack((dx,dy))
                else:
                    d = preprocessing.normalize(d, norm='l1', axis=1)

                if feat_t != 'trj':
                    d = rootSIFT(d)

                if pca_reduction:
                    d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality

                d = np.ascontiguousarray(d, dtype=np.float32)  # required in many of Yael functions

                output_filepath = join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')
                # compute FV of the video
                if not treelike:
                    fv = ynumpy.fisher(cache[feat_t]['gmm'], d, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=fv), f)

                else:  # or separately the FVs of the tree nodes
                    fvtree = dict()
                    if len(clusters['tree']) == 1:
                        fvtree[1] = ynumpy.fisher(cache[feat_t]['gmm'], d, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                    else:
                        T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                        for parent_idx, children_inds in T.iteritems():
                            # (in a global representation)
                            node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                            fvtree[parent_idx] = ynumpy.fisher(cache[feat_t]['gmm'], d[node_inds,:], INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=fvtree), f)

            elapsed_time = time.time() - start_time
            if verbose:
                print('[_compute_fv_descriptors] %s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def _compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                            pca_reduction=False, treelike=True, clusters_path=None, verbose=False):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        # cach'd pca and gmm

        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(join(feats_path, feat_t + '-' + str(k)))
            except OSError:
                pass

        cache = None

        # process videos
        total = len(videonames)
        for i in indices:
            # FV computed for all feature types? see the last in INTERNAL_PARAMETERS['feature_types']
            all_done = np.all([isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl'))
                   for feat_t in feat_types])
            if all_done:
                if verbose:
                    print('[_compute_vd_descriptors] %s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(join(intermediates_path, 'gmm' + ('-' if pca_reduction else '-nopca-') + feat_t + '-' + str(k) + '.pkl'), 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(join(tracklets_path, 'obj', videonames[i] + '.pkl'), 'rb') as f:
                obj = cPickle.load(f)
            with open(join(clusters_path, videonames[i] + '.pkl'), 'rb') as f:
                clusters = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                if isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')):
                    continue

                # load video tracklets' feature
                with open(join(tracklets_path, feat_t, videonames[i] + '.pkl'), 'rb') as f:
                    d = cPickle.load(f)

                if feat_t == 'trj': # (special case)
                    d = convert_positions_to_displacements(d)

                if feat_t == 'mbh':
                    dx = preprocessing.normalize(d[:,:d.shape[1]/2], norm='l1', axis=1)
                    dy = preprocessing.normalize(d[:,d.shape[1]/2:], norm='l1', axis=1)
                    d = np.hstack((dx,dy))
                else:
                    d = preprocessing.normalize(d, norm='l1', axis=1)

                if feat_t != 'trj':
                    d = rootSIFT(d)

                if pca_reduction:
                    d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality

                d = np.ascontiguousarray(d, dtype=np.float32)  # required in many of Yael functions

                output_filepath = join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')
                # compute FV of the video
                if not treelike:
                    # (in a per-frame representation)
                    fids = np.unique(obj[:,0])
                    V = [] # row-wise fisher vectors (matrix)
                    for f in fids:
                        tmp = d[np.where(obj[:,0] == f)[0],:]  # hopefully this is contiguous if d already was
                        fv = ynumpy.fisher(cache[feat_t]['gmm'], tmp, include=INTERNAL_PARAMETERS['fv_repr_feats'])  # f-th frame fisher vec
                        V.append(fv)  # no normalization or nothing (it's done when computing darwin)

                    vd = videodarwin.darwin(np.array(V))

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=vd), f)

                else:  # or separately the FVs of the tree nodes
                    vdtree = dict()
                    if len(clusters['tree']) == 1:
                        fids = np.unique(obj[:,0]).astype('int')
                        U = None
                        for f,fid in enumerate(range(fids[0], fids[-1]+1)):
                            aux_inds = np.where(obj[:,0] == fid)[0]
                            if len(aux_inds) == 0:
                                continue
                            fv = ynumpy.fisher(cache[feat_t]['gmm'], d[aux_inds,:], INTERNAL_PARAMETERS['fv_repr_feats'])
                            if U is None:
                                U = np.zeros((fids[-1]-fids[0]+1, len(fv)), dtype=np.float32)
                            U[f] = fv
                        vdtree[1] = (videodarwin.darwin(U)).astype('float32')
                    else:
                        T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                        for parent_idx, children_inds in T.iteritems():
                            # (in a per-frame representation)
                            node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                            fids = np.unique(obj[node_inds,0]).astype('int')
                            U = None
                            for f,fid in enumerate(range(fids[0], fids[-1]+1)):
                                aux_inds = np.where(obj[node_inds,0] == fid)[0]
                                if len(aux_inds) == 0:
                                    continue
                                fv = ynumpy.fisher(cache[feat_t]['gmm'], d[aux_inds,:], INTERNAL_PARAMETERS['fv_repr_feats'])
                                if U is None:
                                    U = np.zeros((fids[-1]-fids[0]+1, len(fv)), dtype=np.float32)
                                U[f] = fv  # no normalization or nothing (it's done when computing darwin)
                            vdtree[parent_idx] = (videodarwin.darwin(U)).astype('float32')

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=vdtree), f)

            elapsed_time = time.time() - start_time
            if verbose:
                print('[_compute_vd_descriptors] %s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def train_bovw_codebooks(tracklets_path, videonames, traintest_parts, feat_types, intermediates_path, pca_reduction=False, nt=1, verbose=False):
    try:
        makedirs(intermediates_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        train_inds = np.where(part <= 0)[0]  # train codebook for each possible training parition
        total = len(train_inds)
        num_samples_per_vid = int(INTERNAL_PARAMETERS['n_samples'] / float(total))

        # process the videos
        for i, feat_t in enumerate(feat_types):
            output_filepath = join(intermediates_path, 'bovw' + ('-' if pca_reduction else '-nopca-') + feat_t + '-' + str(k) + '.pkl')

            if isfile(output_filepath):
                if verbose:
                    print('[train_bovw_codebooks] %s -> OK' % output_filepath)
                continue

            start_time = time.time()

            D = load_tracklets_sample(tracklets_path, videonames, train_inds, feat_t, num_samples_per_vid, verbose=verbose)

            # (special case) trajectory features are originally positions
            if feat_t == 'trj':
                D = convert_positions_to_displacements(D)

            if feat_t == 'mbh':
                Dx = preprocessing.normalize(D[:,:D.shape[1]/2], norm='l1', axis=1)
                Dy = preprocessing.normalize(D[:,D.shape[1]/2:], norm='l1', axis=1)
                D = np.hstack((Dx,Dy))
            else:
                D = preprocessing.normalize(D, norm='l1', axis=1)

            if feat_t != 'trj':
                D = rootSIFT(D)

            # compute PCA map and reduce dimensionality
            if pca_reduction:
                pca = PCA(n_components=int(INTERNAL_PARAMETERS['reduction_factor']*D.shape[1]), copy=False)
                D = pca.fit_transform(D)

            # train codebook for later BOVW computation
            D = np.ascontiguousarray(D, dtype=np.float32)
            cb = ynumpy.kmeans(D, INTERNAL_PARAMETERS['bovw_codebook_k'], \
                               distance_type=2, nt=nt, niter=100, seed=0, redo=1, \
                               verbose=verbose, normalize=False, init='kmeans++')

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(pca=(pca if pca_reduction else None), codebook=cb), f)

            elapsed_time = time.time() - start_time
            if verbose:
                print('[train_bovw_codebooks] %s -> DONE (in %.2f secs)' % (feat_t, elapsed_time))


def train_fv_gmms(tracklets_path, videonames, traintest_parts, feat_types, intermediates_path, pca_reduction=False, nt=4, verbose=False):
    try:
        makedirs(intermediates_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        train_inds = np.where(np.array(part) <= 0)[0]  # train codebook for each possible training parition
        num_samples_per_vid = int(INTERNAL_PARAMETERS['n_samples'] / float(len(train_inds)))

        # process the videos
        for i, feat_t in enumerate(feat_types):
            D = None

            # Train GMMs
            output_filepath = join(intermediates_path, 'gmm' + ('-' if pca_reduction else '-nopca-') + feat_t + '-' + str(k) + '.pkl')
            if isfile(output_filepath):
                if verbose:
                    print('[train_fv_gmms] %s -> OK' % output_filepath)
                continue

            start_time = time.time()

            D = load_tracklets_sample(tracklets_path, videonames, train_inds, feat_t, num_samples_per_vid, verbose=verbose)

            # (special case) trajectory features are originally positions
            if feat_t == 'trj':
                D = convert_positions_to_displacements(D)

            if feat_t == 'mbh':
                Dx = preprocessing.normalize(D[:,:D.shape[1]/2], norm='l1', axis=1)
                Dy = preprocessing.normalize(D[:,D.shape[1]/2:], norm='l1', axis=1)
                D = np.hstack((Dx,Dy))
            else:
                D = preprocessing.normalize(D, norm='l1', axis=1)

            if feat_t != 'trj':
                D = rootSIFT(D)

            # compute PCA map and reduce dimensionality
            if pca_reduction:
                pca = PCA(n_components=int(INTERNAL_PARAMETERS['reduction_factor']*D.shape[1]), copy=False)
                D = pca.fit_transform(D)

            # train GMMs for later FV computation
            D = np.ascontiguousarray(D, dtype=np.float32)
            gmm = ynumpy.gmm_learn(D, INTERNAL_PARAMETERS['fv_gmm_k'], nt=nt, niter=500, redo=1)

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(pca=(pca if pca_reduction else None), gmm=gmm), f)
            # with open(join(intermediates_path, 'gmm-sample' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl'), 'wb') as f:
            #     cPickle.dump(D,f)

            elapsed_time = time.time() - start_time
            if verbose:
                print('[train_fv_gmms] %s -> DONE (in %.2f secs)' % (feat_t, elapsed_time))



def load_tracklets_sample(tracklets_path, videonames, data_inds, feat_t, num_samples_per_vid, verbose=False):
    D = None  # feat_t's sampled tracklets
    ptr = 0
    for j in range(0, len(data_inds)):
        idx = data_inds[j]

        filepath = join(tracklets_path, feat_t, videonames[idx] + '.pkl')
        if not isfile(filepath):
            sys.stderr.write('# ERROR: missing training instance'
                             ' {}\n'.format(filepath))
            sys.stderr.flush()
            quit()

        with open(filepath, 'rb') as f:
            d = cPickle.load(f)
            if verbose:
                print('[load_tracklets_sample] %s (num feats: %d)' % (filepath, d.shape[1]))

        # init sample
        if D is None:
            D = np.zeros((INTERNAL_PARAMETERS['n_samples'], d.shape[1]), dtype=np.float32)
        # create a random permutation for sampling some tracklets in this vids
        randp = np.random.permutation(d.shape[0])
        if d.shape[0] > num_samples_per_vid:
            randp = randp[:num_samples_per_vid]
        D[ptr:ptr+len(randp),:] = d[randp,:]
        ptr += len(randp)

    return D[:ptr,:] # cut out extra reserved space



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

    # D = np.zeros((P.shape[0], Vx.shape[1]+Vy.shape[1]), dtype=P.dtype)
    # normx = np.linalg.norm(Vx, ord=2, axis=1)[:,np.newaxis]
    # normy = np.linalg.norm(Vy, ord=2, axis=1)[:,np.newaxis]
    # D[:,::2]  = Vx / normx  # l2-normalize
    # D[:,1::2] = Vy / normy

    D = np.zeros((P.shape[0], Vx.shape[1]+Vy.shape[1]), dtype=P.dtype)
    D[:,::2]  = Vx
    D[:,1::2] = Vy

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

def bovw(codebook, X, nt=1):
    inds, dists = ynumpy.knn(X, codebook, nnn=1, distance_type=2, nt=1)
    bins, _ = np.histogram(inds[:,0], bins=INTERNAL_PARAMETERS['bovw_codebook_k'])

    return bins


def rootSIFT(X, p=0.5):
    return np.sign(X) * (np.abs(X) ** p)


def normalize(x, norm='l2',dtype=np.float32):
    if norm == 'l1':
        return x.astype(dtype=dtype) / (np.abs(x)).sum()
    elif norm == 'l2':
        # norms = np.sqrt(np.sum(x ** 2, 1))
        # return x / norms.reshape(-1, 1)
        return x.astype(dtype=dtype) / np.sqrt(np.dot(x,x))
    else:
        raise AttributeError(norm)


