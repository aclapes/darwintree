__author__ = 'aclapes'

from os.path import isfile, exists
from os import makedirs
import cPickle
import random
import time
from math import isnan

import numpy as np
from sklearn.metrics import pairwise

from spectral_division import spectral_embedding_nystrom, spectral_clustering_division

import cv2

def cluster(tracklets_path, videonames, st, num_videos, clusters_path, visualize=False):
    """
    This function implements the method described in Section 2 ("Clustering dense tracklets")
    of the paper 'Activity representation with motion hierarchies' (IJCV, 2014).
    :param tracklets_path:
    :param videonames:
    :param st:
    :param num_videos:
    :param clusters_path:
    :return:
    """
    if not exists(clusters_path):
        makedirs(clusters_path)

    # process the videos
    total = len(videonames)
    for i in range(st,min(st+num_videos,total)):
        if isfile(clusters_path + videonames[i] + '.pkl'):
            print('%s -> OK' % videonames[i])
            continue

        start_time = time.time()

        with open(tracklets_path + 'obj/' + videonames[i] + '.pkl', 'rb') as f:
            data_obj = cPickle.load(f)
        with open(tracklets_path + 'trj/' + videonames[i] + '.pkl', 'rb') as f:
            data_trj = cPickle.load(f)

        # (Sec. 2.2) get a dictionary of separate channels
        D = dict()
        for k in xrange(data_obj.shape[0]): # range(0,100):  #
            T = np.reshape(data_trj[k], (data_trj.shape[1]/2,2))  # trajectory features into matrix (time length x 2)
            D.setdefault('x',[]).append( T[1:,0] )  # x's offset + x's relative displacement
            D.setdefault('y',[]).append( T[1:,1] ) #  y's offset + y's relative displacement
            D.setdefault('t',[]).append( data_obj[k,0] - np.linspace(T.shape[0]-1, 0, T.shape[0]) )
            D.setdefault('v_x',[]).append( T[1:,0] - T[:-1,0] )
            D.setdefault('v_y',[]).append( T[1:,1] - T[:-1,1] )

        # (Sec. 2.3.1)
        # A, B = get_tracklet_similarities(D, data_obj[:,7:9])
        # create a subsample (n << N) stratified by a grid
        insample, outsample = stratified_subsample_of_tracklets_in_grid(data_obj[:,7:9], p=0.01)  # given tracklets (ending) positions
        # get the similarities of
        A, medians = multimodal_product_kernel(D, insample, insample)  # (n), n << N tracklets
        B, _ = multimodal_product_kernel(D, insample, outsample, medians=medians)  # (N - n) tracklets
        # (Sec. 2.3.2 and 2.3.3)
        E_ = spectral_embedding_nystrom(np.hstack((A,B)))
        E = np.zeros(E_.shape, dtype=E_.dtype)
        E[insample,:] = E_[:len(insample),:]
        E[outsample,:] = E_[len(insample):,:]
        # (Sec. 2.4)
        best_labels, int_paths = spectral_clustering_division(E, data_obj[:,7:10], normalize_geoms=False)

        with open(clusters_path + videonames[i] + '.pkl', 'wb') as f:
            cPickle.dump({'best_labels' : best_labels, 'int_paths' : int_paths}, f)

        elapsed_time = time.time() - start_time
        print('%s -> DONE (in %.2f secs)' % (videonames[i], elapsed_time))

        if visualize:
            n_paths = len(np.unique(int_paths))
            print '#DEBUG n_labels:', n_paths

            xres = 528
            yres = 224
            A = np.zeros((yres,1280,3), dtype=np.uint8)
            for l,p in enumerate(np.unique(int_paths)):
                cluster_inds = np.where(int_paths == p)[0]
                hue = (float(l)/n_paths + random.random()) % 1
                for k in xrange(0, len(cluster_inds)):
                    idx = cluster_inds[k]
                    T = np.reshape(data_trj[idx,:], (data_trj.shape[1]/2,2))
                    t = data_obj[idx,9]
                    for j in xrange(1,T.shape[0]):
                        pt1 = (int(T[j-1,0]*xres-xres/2+t*1280), int(T[j-1,1]*yres))
                        pt2 = (int(T[j,0]*xres-xres/2+t*1280), int(T[j,1]*yres))
                        cv2.line(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)), 1)
                    cv2.circle(A, pt2, 1, hsv_to_rgb((hue,0.5,1.0)), -1)
                cv2.imshow("#DEBUG Clustering visualization", A)
                cv2.waitKey(0)


# ==============================================================================
# Helper functions
# ==============================================================================


def stratified_subsample_of_tracklets_in_grid(P, nx=3, ny=3, p=0.01):
    """
    Subsample a factor p of the total tracklets stratifying the sampling in a
    grid of nx-by-ny cells.
    :param P: N-by-2 matrix of tracklet (ending) positions
    :param p: the sampling probability
    :param nx: number of horizontal divisions of the grid
    :param ny: number of vertical divisions of the grid
    :return insample, outsample:
    """
    p_cell = p / (nx*ny)
    insample = []
    outsample = []
    for i in range(0,ny):
        for j in range(0,nx):
            y_ran = (i*(1.0/ny), (i+1)*(1.0/ny))
            x_ran = (j*(1.0/nx), (j+1)*(1.0/nx))
            cell_inds = np.where((P[:,0] >= x_ran[0]) & (P[:,0] < x_ran[1]) & (P[:,1] >= y_ran[0]) & (P[:,1] < y_ran[1]))[0]
            m = len(cell_inds)
            random.seed(74)
            sorted_inds = sorted(np.arange(m, dtype=np.int32), key=lambda k: np.random.random())
            insample.append(np.array(sorted_inds[:int(np.ceil(m*p_cell))], dtype=np.int32))
            outsample.append(np.array(sorted_inds[int(np.ceil(m*p_cell)):], dtype=np.int32))

    return np.concatenate(insample), np.concatenate(outsample)


def multimodal_product_kernel(D, primary_inds=None, secondary_inds=None, medians=None):
    """
    Merges the different modalities (or channels) using the product of rbf kernels.
    The similarity matrix computed is the one from the samples in the primary indices to the secondary indices.
    If some indices are not specified (None) all samples are used.
    :param D: a python dict containing the data in the different modalitites (or channels).
    keys are the names of the modalities
    :param primary_inds:
    :param secondary_inds:
    :return K:
    """
    n = len(primary_inds) if primary_inds is not None else len(D['x'])
    m = len(secondary_inds) if secondary_inds is not None else len(D['x'])

    channels = ['x','y','t','v_x','v_y']
    if medians is None:
        medians = []

    K = np.ones((n, m), dtype=np.float32)  # prepare kernel product
    for i, channel_t in enumerate(channels):
        D[channel_t] = np.array(D[channel_t], dtype=np.float32)
        X_primary = D[channel_t][primary_inds] if primary_inds is not None else D[channel_t]
        X_secondary = D[channel_t][secondary_inds] if secondary_inds is not None else D[channel_t]
        S = pairwise.euclidean_distances(X_primary, X_secondary)
        median = np.nanmedian(S[S!=0])
        if len(medians) < len(channels):
            medians.append(median)
        gamma = 1.0 / (2 * median) if not isnan(median) and median != 0.0 else 0.0
        K = np.multiply(K, np.exp(-gamma * np.power(S,2))) # rbf kernel and element-wise multiplication
    return K, medians


def get_tracklet_similarities(D, P):
    """
    Creates A and B.
    :param D: dictionary of modalities, each modality is N-by-{#features}
    :param P: matrix 2-by-N
    :return A, B:
    """
    # create a subsample (n << N) stratified by a grid
    insample, outsample = stratified_subsample_of_tracklets_in_grid(P, p=0.01)  # given tracklets (ending) positions
    # get the similarities of
    A, medians = multimodal_product_kernel(D, insample, insample)  # (n), n << N tracklets
    B, _ = multimodal_product_kernel(D, insample, outsample, medians=medians)  # (N - n) tracklets

    return A, B





def hsv_to_rgb(hsv):
    '''
    HSV values in [0..1]
    :param h:
    :param s:
    :param v:
    :return (r, g, b) tuple, with values from 0 to 255:
    '''
    h, s, v = hsv[0], hsv[1], hsv[2]

    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return (int(r*256), int(g*256), int(b*256))