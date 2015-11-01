__author__ = 'aclapes'

import numpy as np
import subprocess
from sklearn.neighbors import KDTree
import time
import cPickle
from os.path import isfile, exists
from os import makedirs
import fileinput
from spectral_division import build_geom_neighbor_graph
import pyflann

# some hard-coded constants
FEATURE_EXTRACTOR_RELPATH = 'release/'

INTERNAL_PARAMETERS = dict(
    L = 5,
    nfeats = dict(
        obj = 10,
        trj = 2,
        hog = 96,
        hof = 108,
        mbh = 192
    ),
    indices = dict(
        meanx = 1,
        meany = 2
    )
)

def extract(fullvideonames, videonames, st, num_videos, feat_types, tracklets_path):
    L = INTERNAL_PARAMETERS['L']
    # establish the features and their dimensions' start-end
    feats_beginend = {'obj' : (0,               \
                              INTERNAL_PARAMETERS['nfeats']['obj']), \
                     'trj' : (INTERNAL_PARAMETERS['nfeats']['obj'],                           \
                              INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))),   \
                     'hog' : (INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1)),                 \
                              INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))+INTERNAL_PARAMETERS['nfeats']['hog']), \
                     'hof' : (INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))+INTERNAL_PARAMETERS['nfeats']['hog'],                 \
                              INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))+INTERNAL_PARAMETERS['nfeats']['hog']+INTERNAL_PARAMETERS['nfeats']['hof']), \
                     'mbh': (INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))+INTERNAL_PARAMETERS['nfeats']['hog']+INTERNAL_PARAMETERS['nfeats']['hof'],                  \
                             INTERNAL_PARAMETERS['nfeats']['obj']+(INTERNAL_PARAMETERS['nfeats']['trj']*(L+1))+INTERNAL_PARAMETERS['nfeats']['hog']+INTERNAL_PARAMETERS['nfeats']['hof']+INTERNAL_PARAMETERS['nfeats']['mbh'])}

    # prepare output directories
    if not exists('tmpfiles/'):
        makedirs('tmpfiles/')
    if not exists(tracklets_path):
        makedirs(tracklets_path)
    for feat_t in feats_beginend.keys():
        if not exists(tracklets_path + feat_t + '/'):
            makedirs(tracklets_path + feat_t + '/')

    # process the videos
    total = len(fullvideonames)
    for i in range(st,min(st+num_videos,total)):
        if isfile(tracklets_path + 'obj/' + videonames[i] + '.pkl'):
            print('%s.avi -> OK' % fullvideonames[i])
            continue

        start_time = time.time()
        # extract the features into temporary file
        if not isfile('tmpfiles/' + videonames[i] + '.tmp'):
            extract_wang_features(fullvideonames[i] + '.avi', L, 'tmpfiles/' + videonames[i] + '.tmp')

        # read the temporary file to numpy array
        data = []
        for line in fileinput.input(['tmpfiles/' + videonames[i] + '.tmp']):
            data.append(np.array(line.strip().split('\t'), dtype=np.float32))
        data = np.array(data, dtype=np.float32)

        # filter low density tracklets
        inliers = filter_low_density(data)

        # store feature types separately
        for feat_t in feats_beginend.keys():
            with open(tracklets_path + feat_t + '/' + videonames[i] + '.pkl','wb') as f:
                cPickle.dump(data[inliers, feats_beginend[feat_t][0]:feats_beginend[feat_t][1]], f)

        elapsed_time = time.time() - start_time
        print('%s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))


# ==============================================================================
# Helper functions
# ==============================================================================


# Version using precomputed optical flow (stored in .flo files)
def extract_wang_features(videofile_path, traj_length, output_features_path):
    ''' Use external program (DenseTrack) to extract the features '''
    argsArray = ['./DenseTrackStab', videofile_path, \
                 '-L', str(traj_length)]  # DenseTrackStab is not accepting parameters, hardcoded the L in there

    f = open(output_features_path,"wb")
    proc = subprocess.Popen(' '.join(argsArray), cwd=FEATURE_EXTRACTOR_RELPATH, shell=True, stdout=f)
    proc.communicate()
    f.close()

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

            if k+1 <= len(subset_indices):
                dists, inds = tree.query(P[i,:],k=k+1)
                dists = dists[0,1:]  # asked the neighbors of only 1 instance, returned in dists as 0-th element
            else:  #len(subset_indices) > 1:
                dists, inds = tree.query(P[i,:],k=len(subset_indices))
                dists = np.concatenate([dists[0,1:], [np.nan]*(k-len(dists[0,1:]))])
            all_sparsities[i,:] = dists

    local_sparsities = np.nanmean(all_sparsities, axis=1)
    mean_sparsity = np.nanmean(all_sparsities)
    stddev_sparsity = np.nanstd(all_sparsities)
    inliers = np.where(local_sparsities <= (mean_sparsity + stddev_sparsity))[0]

    return inliers

