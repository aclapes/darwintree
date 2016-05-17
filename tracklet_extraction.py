__author__ = 'aclapes'

import numpy as np
import subprocess
from sklearn.neighbors import KDTree
import time
import cPickle
from os.path import isfile, exists, join
from os import makedirs
import fileinput
from spectral_division import build_geom_neighbor_graph
import pyflann
from joblib import delayed, Parallel
import sys
import warnings

# some hard-coded constants
FEATURE_EXTRACTOR_RELPATH = 'release/'

INTERNAL_PARAMETERS = dict(
    L = 15,
    feats_dict = dict(
        obj = 10,
        trj = 2,  # 2-by-L actually
        hog = 96,
        hof = 108,
        mbh = 192
    ),
    indices = dict(
        meanx = 1,
        meany = 2
    )
)

def extract(fullvideonames, videonames, feat_types, traj_length, tracklets_path, nt=4, verbose=False):
    inds = np.random.permutation(len(videonames)).astype('int')
    # inds = np.linspace(0,len(videonames)-1,len(videonames)).astype('int')
    # step = np.int(np.floor(len(inds)/nt)+1)
    #inds[i*step:((i+1)*step if (i+1)*step < len(inds) else len(inds))],
    Parallel(n_jobs=nt, backend='threading')(delayed(_extract)(fullvideonames, videonames, [i], \
                                                               feat_types, traj_length, tracklets_path, verbose=verbose)
                                             for i in inds)


def _extract(fullvideonames, videonames, indices, feat_types, traj_length, tracklets_path, verbose=False):
    """
    Extract features using Improved Dense Trajectories by Wang et. al.
    :param fullvideonames:
    :param videonames:
    :param indices:
    :param feat_types:
    :param tracklets_path:
    :return:
    """
    feats_beginend = get_features_beginend(INTERNAL_PARAMETERS['feats_dict'], traj_length)

    # prepare output directories

    try:
        makedirs(tracklets_path)
    except OSError:
        pass

    try:
        makedirs(join(tracklets_path, 'tmp'))
    except OSError:
        pass

    for feat_t in feats_beginend.keys():
        try:
            makedirs(join(tracklets_path, feat_t))
        except OSError:
            pass

    # process the videos
    total = len(fullvideonames)
    for i in indices:
        all_done = np.all([isfile(join(tracklets_path, feat_t, videonames[i] + '.pkl'))
                           for feat_t in feats_beginend.keys()])
        if all_done:
            if verbose:
                print('[_extract] %s -> OK' % fullvideonames[i])
            continue

        start_time = time.time()
        # extract the features into temporary file
        tracklets_filepath = join(tracklets_path, 'tmp/', videonames[i] + '.dat')
        if not isfile(tracklets_filepath):
            extract_wang_features(fullvideonames[i], tracklets_filepath, traj_length=traj_length)

        # read the temporary file to numpy array
        finput = fileinput.FileInput(tracklets_filepath)
        data = []
        for line in finput:
            row = np.array(line.strip().split('\t'), dtype=np.float32)
            data.append(row)
        finput.close()

        try:
            data = np.vstack(data)
        except ValueError:
            # empty row
            sys.stderr.write("[_extract] Error reading tracklets file: " + tracklets_filepath + '\n')
            sys.stderr.flush()
            continue

        # filter low density tracklets
        inliers = filter_low_density(data)

        # store feature types separately
        for feat_t in feats_beginend.keys():
            with open(join(tracklets_path, feat_t, videonames[i] + '.pkl'),'wb') as f:
                cPickle.dump(data[:, feats_beginend[feat_t][0]:feats_beginend[feat_t][1]], f)  # TODO: : -> inliners

        elapsed_time = time.time() - start_time
        if verbose:
            print('[_extract] %s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


# ==============================================================================
# Helper functions
# ==============================================================================

def get_features_beginend(feats_dict, L):
    # establish the features and their dimensions' start-end
    feats_beginend = {'obj' : (0,                 \
                              feats_dict['obj']), \
                     'trj' : (feats_dict['obj'],                                        \
                              feats_dict['obj']+(feats_dict['trj']*L)),   \
                     'hog' : (feats_dict['obj']+(feats_dict['trj']*L),                              \
                              feats_dict['obj']+(feats_dict['trj']*L)+feats_dict['hog']), \
                     'hof' : (feats_dict['obj']+(feats_dict['trj']*L)+feats_dict['hog'],                              \
                              feats_dict['obj']+(feats_dict['trj']*L)+feats_dict['hog']+feats_dict['hof']), \
                     'mbh' : (feats_dict['obj']+(feats_dict['trj']*L)+feats_dict['hog']+feats_dict['hof'],                           \
                              feats_dict['obj']+(feats_dict['trj']*L)+feats_dict['hog']+feats_dict['hof']+feats_dict['mbh'])}
    return feats_beginend

# Version using precomputed optical flow (stored in .flo files)
def extract_wang_features(videofile_path, output_features_path, traj_length=15):
    ''' Use external program (DenseTrack) to extract the features '''
    argsArray = ['./DenseTrackStab', videofile_path, \
                 '-L', str(traj_length)]  # DenseTrackStab is not accepting parameters, hardcoded the L in there

    try:
        f = open(output_features_path,'wb')
        proc = subprocess.Popen(' '.join(argsArray), cwd=FEATURE_EXTRACTOR_RELPATH, shell=True, stdout=f)
        proc.communicate()
        f.close()
    except IOError:
        sys.stderr.write('[Error] Cannot open file for writing: %s\n' % videofile_path)


def filter_low_density(data, k=30, r=5):
    """
    Filter out low density tracklets from the sequence.
    :param data: the tracklets, a T x num_features matrix.
    :return:
    """

    # each tracklet's mean x and y position
    P = data[:,[INTERNAL_PARAMETERS['indices']['meanx'],INTERNAL_PARAMETERS['indices']['meany']]]  # (these are index 1 and 2 of data)

    all_sparsities = np.zeros((P.shape[0],k), dtype=np.float32)
    subset_indices = []  # optimization. see (*) below
    for i in range(0, P.shape[0]):
        new_subset_indices = np.where((data[:,0] >= data[i,0] - r) & (data[:,0] <= data[i,0] + r))[0]
        if len(new_subset_indices) == 1:
            all_sparsities[i,:] = np.nan
        else:
            # (*) a new KDTree is constructed only if the subset of data changes
            if not np.array_equal(new_subset_indices, subset_indices):
                subset_indices = new_subset_indices
                tree = KDTree(P[subset_indices,:], leaf_size=1e3)

            p = P[i,:].reshape(1,-1)  # query instance
            if k+1 <= len(subset_indices):
                dists, inds = tree.query(p, k=k+1)
                dists = dists[0,1:]  # asked the neighbors of only 1 instance, returned in dists as 0-th element
            else:  #len(subset_indices) > 1:
                dists, inds = tree.query(p, k=len(subset_indices))
                dists = np.concatenate([dists[0,1:], [np.nan]*(k-len(dists[0,1:]))])
            all_sparsities[i,:] = dists

    local_sparsities = np.nanmean(all_sparsities, axis=1)
    mean_sparsity = np.nanmean(all_sparsities)
    stddev_sparsity = np.nanstd(all_sparsities)

    f = 1.0
    while f <= 3.0:
        inliers = np.where(local_sparsities <= (mean_sparsity + f * stddev_sparsity))[0]
        if len(inliers) > 0: # all ok
            return inliers
        f += 1.0

    return np.where(~np.isnan(local_sparsities))[0]
