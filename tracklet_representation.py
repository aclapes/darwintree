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
    n_samples = 1000000,  # TODO: set to 1000000
    reduction_factor = 0.5,   # keep after a fraction of the dimensions after applying pca
    # bulding codebooks
    bovw_codebook_k = 4000,
    bovw_lnorm = 1,
    # building GMMs
    fv_gmm_k = 256,  # number of gaussian components
    fv_repr_feats = ['mu','sigma']
)


def compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                             pca_reduction=False, treelike=True, clusters_path=None):
    _compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, np.arange(len(videonames)), feat_types, feats_path, \
                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)

def compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                           pca_reduction=False, treelike=True, clusters_path=None):
    _compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, np.arange(len(videonames)), feat_types, feats_path, \
                            pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)

def compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                           pca_reduction=False, treelike=True, clusters_path=None):
    _compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, np.arange(len(videonames)), feat_types, feats_path, \
                            pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)


def compute_bovw_descriptors_multiprocess(tracklets_path, intermediates_path, videonames, traintest_parts, st, num_videos, feat_types, feats_path, \
                                          pca_reduction=False, treelike=True, clusters_path=None):
    inds = np.linspace(st, st+num_videos-1, num_videos)
    _compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, inds, feat_types, feats_path, \
                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)

def compute_fv_descriptors_multiprocess(tracklets_path, intermediates_path, videonames, traintest_parts, st, num_videos, feat_types, feats_path, \
                                        pca_reduction=False, treelike=True, clusters_path=None):
    inds = np.linspace(st, st+num_videos-1, num_videos)
    _compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, inds, feat_types, feats_path, \
                            pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)

def compute_vd_descriptors_multiprocess(tracklets_path, intermediates_path, videonames, traintest_parts, st, num_videos, feat_types, feats_path, \
                                        pca_reduction=False, treelike=True, clusters_path=None):
    inds = np.linspace(st, st+num_videos-1, num_videos)
    _compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, inds, feat_types, feats_path, \
                            pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)


def compute_bovw_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                         nt=4, pca_reduction=False, treelike=True, clusters_path=None):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_bovw_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                                [i], feat_types, feats_path, \
                                                                                pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)
                                                           for i in xrange(len(videonames)))

def compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                       nt=4, pca_reduction=False, treelike=True, clusters_path=None):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_fv_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                              [i], feat_types, feats_path, \
                                                                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)
                                                           for i in xrange(len(videonames)))

def compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, feat_types, feats_path, \
                                       nt=4, pca_reduction=False, treelike=True, clusters_path=None):
    Parallel(n_jobs=nt, backend='threading')(delayed(_compute_vd_descriptors)(tracklets_path, intermediates_path, videonames, traintest_parts, \
                                                                              [i], feat_types, feats_path, \
                                                                              pca_reduction=pca_reduction, treelike=treelike, clusters_path=clusters_path)
                                                           for i in xrange(len(videonames)))


# ==============================================================================
# Main functions
# ==============================================================================


def _compute_bovw_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                              pca_reduction=False, treelike=True, clusters_path=None):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        # cach'd pca and gmm
        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(feats_path + feat_t + '-' + str(k))
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
                # for j, feat_t in enumerate(feat_types):
                #     featnames.setdefault(feat_t, []).append(feats_path + feat_t + '/' + videonames[i] + '-fvtree.pkl')
                print('%s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(intermediates_path + 'bovw' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(tracklets_path + 'obj/' + videonames[i] + '.pkl', 'rb') as f:
                obj = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                # load video tracklets' feature
                with open(tracklets_path + feat_t + '/' + videonames[i] + '.pkl', 'rb') as f:
                    d = cPickle.load(f)
                    if feat_t == 'trj': # (special case)
                        d = convert_positions_to_displacements(d)

                # pre-processing
                d = rootSIFT(preprocessing.normalize(d, norm='l1', axis=1))  # section 3.1 from "improved dense trajectories)
                if pca_reduction:
                    d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality

                output_filepath = join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')
                # compute BOVW of the video
                if not treelike:
                    b = bovw(cache[feat_t]['codebook'], d)
                    b = preprocessing.normalize(b, norm='l1')
                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=b), f)

                else:  # or separately the BOVWs of the tree nodes
                    with open(clusters_path + videonames[i] + '.pkl', 'rb') as f:
                        clusters = cPickle.load(f)

                    T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                    bovwtree = dict()
                    for parent_idx, children_inds in T.iteritems():
                        # (in a global representation)
                        node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                        b = bovw(cache[feat_t]['codebook'], d[node_inds,:])  # bovw vec
                        bovwtree[parent_idx] = normalize(b, norm='l1')

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=bovwtree), f)

            elapsed_time = time.time() - start_time
            print('%s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def _compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                            pca_reduction=False, treelike=True, clusters_path=None):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        # cach'd pca and gmm

        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(feats_path + feat_t + '-' + str(k))
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
                # for j, feat_t in enumerate(feat_types):
                #     featnames.setdefault(feat_t, []).append(feats_path + feat_t + '/' + videonames[i] + '-fvtree.pkl')
                print('%s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(intermediates_path + 'gmm' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(tracklets_path + 'obj/' + videonames[i] + '.pkl', 'rb') as f:
                obj = cPickle.load(f)
            with open(clusters_path + videonames[i] + '.pkl', 'rb') as f:
                clusters = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                if isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')):
                    continue

                # load video tracklets' feature
                with open(tracklets_path + feat_t + '/' + videonames[i] + '.pkl', 'rb') as f:
                    d = cPickle.load(f)
                    if feat_t == 'trj': # (special case)
                        d = convert_positions_to_displacements(d)

                # pre-processing
                d = rootSIFT(preprocessing.normalize(d, norm='l1', axis=1))  # https://hal.inria.fr/hal-00873267v2/document

                if pca_reduction:
                    d = cache[feat_t]['pca'].transform(d)  # reduce dimensionality

                d = np.ascontiguousarray(d, dtype=np.float32)  # required in many of Yael functions

                output_filepath = join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')
                # compute FV of the video
                if not treelike:
                    fv = ynumpy.fisher(cache[feat_t]['gmm'], d, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                    fv = normalize(rootSIFT(fv,p=0.5), norm='l2')
                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=fv), f)

                else:  # or separately the FVs of the tree nodes
                    fvtree = dict()
                    if len(clusters['tree']) == 1:
                        fv = ynumpy.fisher(cache[feat_t]['gmm'], d, INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                        fvtree[1] = normalize(rootSIFT(fv,p=0.5), norm='l2')
                    else:
                        T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                        for parent_idx, children_inds in T.iteritems():
                            # (in a global representation)
                            node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                            fv = ynumpy.fisher(cache[feat_t]['gmm'], d[node_inds,:], INTERNAL_PARAMETERS['fv_repr_feats'])  # fisher vec
                            fvtree[parent_idx] = normalize(rootSIFT(fv,p=0.5), norm='l2')  # https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=fvtree), f)

            elapsed_time = time.time() - start_time
            print('%s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def _compute_vd_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, indices, feat_types, feats_path, \
                            pca_reduction=False, treelike=True, clusters_path=None):
    try:
        makedirs(feats_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        # cach'd pca and gmm

        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(feats_path + feat_t + '-' + str(k))
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
                # for j, feat_t in enumerate(feat_types):
                #     featnames.setdefault(feat_t, []).append(feats_path + feat_t + '/' + videonames[i] + '-fvtree.pkl')
                print('%s -> OK' % videonames[i])
                continue

            if cache is None:
                cache = dict()
                for j, feat_t in enumerate(feat_types):
                    with open(intermediates_path + 'gmm' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'rb') as f:
                        cache[feat_t] = cPickle.load(f)

            start_time = time.time()

            # object features used for the per-frame FV representation computation (cach'd)
            with open(tracklets_path + 'obj/' + videonames[i] + '.pkl', 'rb') as f:
                obj = cPickle.load(f)
            with open(clusters_path + videonames[i] + '.pkl', 'rb') as f:
                clusters = cPickle.load(f)

            for j, feat_t in enumerate(feat_types):
                if isfile(join(feats_path, feat_t + '-' + str(k), videonames[i] + '.pkl')):
                    continue

                # load video tracklets' feature
                with open(tracklets_path + feat_t + '/' + videonames[i] + '.pkl', 'rb') as f:
                    d = cPickle.load(f)
                    if feat_t == 'trj': # (special case)
                        d = convert_positions_to_displacements(d)

                # pre-processing
                d = rootSIFT(preprocessing.normalize(d, norm='l1', axis=1))  # https://hal.inria.fr/hal-00873267v2/document

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

                    vd = normalize(videodarwin.darwin(np.array(V)))

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(v=vd), f)

                else:  # or separately the FVs of the tree nodes
                    vdtree = dict()
                    if len(clusters['tree']) == 1:
                        fids = np.unique(obj[:,0])
                        V = [ynumpy.fisher(cache[feat_t]['gmm'], d[np.where(obj[:,0] == f)[0],:], INTERNAL_PARAMETERS['fv_repr_feats'])
                             for f in fids]
                        vdtree[1] = normalize(videodarwin.darwin(np.array(V)))
                    else:
                        T = reconstruct_tree_from_leafs(np.unique(clusters['int_paths']))
                        for parent_idx, children_inds in T.iteritems():
                            # (in a per-frame representation)
                            node_inds = np.where(np.any([clusters['int_paths'] == idx for idx in children_inds], axis=0))[0]
                            fids = np.unique(obj[node_inds,0])
                            # dim = INTERNAL_PARAMETERS['fv_gmm_k'] * len(INTERNAL_PARAMETERS['fv_repr_feats']) * d.shape[1]
                            V = []
                            for f in fids:
                                tmp = d[np.where(obj[node_inds,0] == f)[0],:]
                                fv = ynumpy.fisher(cache[feat_t]['gmm'], tmp, INTERNAL_PARAMETERS['fv_repr_feats'])
                                V.append(fv)  # no normalization or nothing (it's done when computing darwin)
                            vdtree[parent_idx] = normalize(videodarwin.darwin(np.array(V)))

                    with open(output_filepath, 'wb') as f:
                        cPickle.dump(dict(tree=vdtree), f)

            elapsed_time = time.time() - start_time
            print('%s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


def train_bovw_codebooks(tracklets_path, videonames, traintest_parts, feat_types, intermediates_path, pca_reduction=False, nt=1):
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
            output_filepath = intermediates_path + 'bovw' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl'
            if isfile(output_filepath):
                print('%s -> OK' % output_filepath)
                continue

            start_time = time.time()

            D = load_tracklets_sample(tracklets_path, videonames, train_inds, feat_t, num_samples_per_vid)

            # (special case) trajectory features are originally positions
            if feat_t == 'trj':
                D = convert_positions_to_displacements(D)

            D = rootSIFT(preprocessing.normalize(D, norm='l1', axis=1))

            # compute PCA map and reduce dimensionality
            if pca_reduction:
                pca = PCA(n_components=int(INTERNAL_PARAMETERS['reduction_factor']*D.shape[1]), copy=False)
                D = pca.fit_transform(D)

            # train codebook for later BOVW computation
            D = np.ascontiguousarray(D, dtype=np.float32)
            cb = ynumpy.kmeans(D, INTERNAL_PARAMETERS['bovw_codebook_k'], \
                               distance_type=2, nt=nt, niter=100, seed=0, redo=1, \
                               verbose=True, normalize=False, init='kmeans++')

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(pca=(pca if pca_reduction else None), codebook=cb), f)

            elapsed_time = time.time() - start_time
            print('%s -> DONE (in %.2f secs)' % (feat_t, elapsed_time))


def train_fv_gmms(tracklets_path, videonames, traintest_parts, feat_types, intermediates_path, pca_reduction=False, nt=4):
    try:
        makedirs(intermediates_path)
    except OSError:
        pass

    for k, part in enumerate(traintest_parts):
        train_inds = np.where(part <= 0)[0]  # train codebook for each possible training parition
        num_samples_per_vid = int(INTERNAL_PARAMETERS['n_samples'] / float(len(train_inds)))

        # process the videos
        for i, feat_t in enumerate(feat_types):
            D = None

            # Train GMMs
            output_filepath = intermediates_path + 'gmm' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl'
            if isfile(output_filepath):
                print('%s -> OK' % output_filepath)
            else:
                start_time = time.time()

                D = load_tracklets_sample(tracklets_path, videonames, train_inds, feat_t, num_samples_per_vid)

                # (special case) trajectory features are originally positions
                if feat_t == 'trj':
                    D = convert_positions_to_displacements(D)

                # scale (rootSIFT)
                D = rootSIFT(preprocessing.normalize(D, norm='l1', axis=1))

                # compute PCA map and reduce dimensionality
                if pca_reduction:
                    pca = PCA(n_components=int(INTERNAL_PARAMETERS['reduction_factor']*D.shape[1]), copy=False)
                    D = pca.fit_transform(D)

                # train GMMs for later FV computation
                D = np.ascontiguousarray(D, dtype=np.float32)
                gmm = ynumpy.gmm_learn(D, INTERNAL_PARAMETERS['fv_gmm_k'], nt=nt, niter=250, redo=1)

                with open(output_filepath, 'wb') as f:
                    cPickle.dump(dict(pca=(pca if pca_reduction else None), gmm=gmm), f)
                with open(intermediates_path + 'gmm-sample' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'wb') as f:
                    cPickle.dump(D,f)

                elapsed_time = time.time() - start_time
                print('%s -> DONE (in %.2f secs)' % (feat_t, elapsed_time))

            # Dimensionality reduction
            # gmm_model_filepath = output_filepath
            # output_filepath = intermediates_path + 'gmm-' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl'
            # if isfile(output_filepath):
            #     print('%s -> OK' % output_filepath)
            # else:
            #     with open(gmm_model_filepath, 'rb') as f:
            #         gmm_model = cPickle.load(f)['gmm']
            #
            #     if D is None:
            #         try:
            #             with open(intermediates_path + 'gmm-sample' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'rb') as f:
            #                 D = cPickle.load(f)
            #         except:
            #             V = [None] * len(train_inds)
            #             for j,idx in enumerate(train_inds):
            #                 print("[train_fv_gmms] PCA sample generation: %d out of %d\n" % (j, len(train_inds)))
            #                 d = None
            #                 try:
            #                     with open(tracklets_path + feat_t + '/' + videonames[idx] + '.pkl', 'rb') as f:
            #                         d = cPickle.load(f)
            #                 except IOError:
            #                     continue
            #                 V[j] = ynumpy.fisher(gmm_model, np.ascontiguousarray(d, dtype=np.float32), include=INTERNAL_PARAMETERS['fv_repr_feats'])
            #
            #             with open(intermediates_path + 'gmm-sample' + ('_pca-' if pca_reduction else '-') + feat_t + '-' + str(k) + '.pkl', 'wb') as f:
            #                 cPickle.dump(V,f)


def load_tracklets_sample(tracklets_path, videonames, data_inds, feat_t, num_samples_per_vid):
    D = None  # feat_t's sampled tracklets
    ptr = 0
    for j in range(0, len(data_inds)):
        idx = data_inds[j]

        filepath = tracklets_path + feat_t + '/' + videonames[idx] + '.pkl'
        if not isfile(filepath):
            sys.stderr.write('# ERROR: missing training instance'
                             ' {}\n'.format(filepath))
            sys.stderr.flush()
            quit()

        with open(filepath, 'rb') as f:
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


