"""Tree structure and hierarchical divisive algorithm for spectral clustering

Used in the paper:

@article{Gaidon2014,
author = {Gaidon, Adrien and Harchaoui, Zaid and Schmid, Cordelia},
title = {{Activity representation with motion hierarchies}},
journal = {IJCV},
year = {2014}
}

LICENSE: BSD

Copyrights: Adrien Gaidon, 2012-2014

"""


import sys
import heapq

import numpy as np
from scipy import sparse
from scipy.sparse.sparsetools import cs_graph_components

import pyflann
from sklearn.cluster import MiniBatchKMeans, KMeans


# The fixed internal paramaters for clustering
INTERNAL_PARAMETERS = dict(
    # generic ones
    min_tube_size=100,   # minimum points per cluster
    max_tube_size=2000,  # maximum points per cluster
    min_k=2,             # lower limit on number of tubes per video
    # build_sym_geom_adjacency
    min_geom_neighbors=10,  # minimum number of geometrical neighbors
    # spectral_clustering_division
    n_threshs=10,  # number of evenly-spaced thresholds to try
    max_depth=62,  # maximum depth for cluster-trees (-> max nodes = 2**h - 1)
    min_evect_amplitude=1e-10,  # min amplitude of proj on eigenvector to split
)


def spectral_embedding_nystrom(AB, ridge=1e-10, nvec=2, copy=True):
    """Approximate spectral embedding using the Nystrom approximation

    Parameters
    ----------
    AB: (n, n+m) array,
        similarities between n sub-sampled points and all n+m points
        (AB = [A; B], where A is assumed p.d.)

    ridge: float,
           small offset added to the diagonal of A for numerical stability

    nvec: int, optional, default: 2,
          number of embedding vectors to use (output dimensionality, nvec < n)

    copy: boolean, optional, default: True,
          work on a copy of AB or not

    Returns
    -------
    E: (n+m, nvec) array,
       the spectral embedding of all points

    Raises
    ------
    IndefiniteError: if A = AB[:n, :n] is not positive-definite

    Notes
    -----
    - One shot-technique from [1]: assumes A is p.d.
      Note, that [1] has some mistakes that are corrected here.

    - Cost in memory is at most 4 time the memory size of AB.

    References
    ----------
    [1] Spectral grouping using the Nystrom method,
        Fowlkes, C. and Belongie, S. and Chung, F. and Malik, J.
        PAMI 2004
    """
    if copy:
        AB = AB.copy()  # XXX memory bottleneck
    n = AB.shape[0]
    #m = AB.shape[1] - n
    assert nvec < n, "Too large number of embedding vectors (%d >= %d)" % (
        nvec, n)
    # make views of the blocks
    A = AB[:, :n]
    B = AB[:, n:]
    # add a ridge for numerical stability as A is generally badly-conditionned
    # XXX use QR decompostion of A for num stab (cf. stable GP)?
    A[np.diag_indices_from(A)] += ridge
    # normalize the components of AB
    b_r = B.sum(axis=1)
    pinvA = spd_pinv(A, check_stability=False)
    d1 = A.sum(axis=1) + b_r
    d2 = np.abs(B.sum(axis=0) + np.dot(np.dot(b_r, pinvA), B))
    # Note: abs not required except when numerical problems
    if np.any(d1 <= 0):
        raise ValueError("numerical issue: negative or null d1 entries")
    if np.any(d2 <= 0):
        raise ValueError("numerical issue: negative or null d2 entries")
    dhat = np.sqrt(1.0 / np.r_[d1, d2])[:, np.newaxis]
    A *= np.dot(dhat[:n], dhat[:n].T)
    B *= np.dot(dhat[:n], dhat[n:].T)
    # square root of the pseudo-inverse
    Asi = spd_pinv(A, square_root=True, check_stability=False)
    # compute the embedding vectors
    AsiB = np.dot(Asi, B)  # XXX time & memory bottleneck (20% of total)
    S = A + np.dot(AsiB, AsiB.T)  # XXX bottleneck (20% of total)
    QS, deltaS = None, None
    for _i in range(4):
        try:
            QS, deltaS, _ = np.linalg.svd(S)
            break
        except np.linalg.LinAlgError:
            _qridge = ridge * 10 ** _i
            S[np.diag_indices_from(S)] += _qridge
            sys.stderr.write(
                "WARNING: SVD didn't converge:"
                "added ridge {0:0.1e} to weird S matrix\n".format(_qridge))
    if QS is None or deltaS is None:
        raise ValueError("numerical issue: ridge too low or weird S")
    if np.any(deltaS <= 0):
        raise ValueError("numerical issue: negative or null deltaS entry")
    _VT = np.dot(np.diag(1.0 / np.sqrt(deltaS)), np.dot(QS.T, Asi))
    VT = np.dot(_VT, AB)  # XXX time & memory bottleneck (20% of total)
    # return the first nvec embedding vectors
    if np.any(VT[0] == 0):
        sys.stderr.write(
            "WARNING: numerical issue: null first eigenvector entries\n")
        # replace 0 entries by the mean
        _m = np.mean(VT[0])
        if _m == 0:
            raise ValueError('numerical issue: first eigenvector is 0')
        VT[0, VT[0] == 0] = _m
    E = VT[1:nvec + 1] / VT[0][np.newaxis, :]
    # small check
    E = np.asarray_chkfinite(E.T)
    return E


def spectral_clustering_division(E, geoms, split_type="threshold", normalize_geoms=True):
    """Divisive hierarchical clustering + model selection

    Recursively split in two by thresholding the eigenvectors in increasing
    eigenvalue order (starting from the second smallest), until we get too small
    tubes, then perform model selection to determine the optimal splits.

    Parameters
    ----------
    E: (n_pts, n_vec), array,
       the spectral embedding of the points on the n_vec smallest eigen-vectors
       (from the second smallest eigen-value)

    geoms: (n_pts, 3) array,
           array of global (x, y, t) positions of the point tracks

    split_type: 'kmeans' or 'threshold' (default),
                the bi-partitioning algorithm used to split nodes

    Returns
    -------
    best_labels: (n_pts, ) array,
                 the found cluster memberships

    int_paths: (n_pts, ) array,
               use np.binary_repr(int_paths[i]) to get the string path of sample i
               Note: root is the left-most '1', outliers have path 0
    """
    global INTERNAL_PARAMETERS
    n_pts, n_vec = E.shape
    _n, _d = geoms.shape
    assert _n == n_pts and _d == 3, "Invalid geoms (%s)" % (str(geoms.shape))
    # limit on tube sizes
    mts = int(INTERNAL_PARAMETERS['min_tube_size'])
    Mts = int(INTERNAL_PARAMETERS['max_tube_size'])
    # lower limit on the number of clusters
    min_n_clusters = int(INTERNAL_PARAMETERS['min_k'])
    # max allowed node depth
    max_depth = int(INTERNAL_PARAMETERS['max_depth'])
    # min eigenvector amplitude for split
    min_evect_amplitude = float(INTERNAL_PARAMETERS['min_evect_amplitude'])
    # number of thresholds to try when using thresholding splits
    n_threshs = int(INTERNAL_PARAMETERS['n_threshs'])

    # check degenerate case: just issue a warning and lower mts
    if n_pts <= 2 * min_n_clusters * mts:
        n_mts = int(max(1, n_pts / (2.0 * min_n_clusters)))
        sys.stderr.write("WARNING: small video" +
                         "({} <= {}) ".format(n_pts, 2 * min_n_clusters * mts) +
                         ": changing min_leaf_size to {}.\n".format(n_mts))
        mts = n_mts

    # get the normalized spatio-temporal positions
    if normalize_geoms:
        nrlz = np.array([640., 480., 1e2])
        ngeoms = geoms.astype(np.float) / nrlz[np.newaxis, :]
        ngeoms -= ngeoms.mean(axis=0)[np.newaxis, :]
    else:
        ngeoms = geoms - geoms.mean(axis=0)[np.newaxis, :]

    # initialize the tree structure
    stree = SpectralTree(
        E, ngeoms, mts, Mts, min_n_clusters, max_depth, min_evect_amplitude,
        split_type, n_threshs)
    # recursively split the leaves in depth-first left-to-right order
    stree.build()

    return stree.labels, stree.int_paths


# ==============================================================================
# Helper functions
# ==============================================================================


def spd_pinv(a, rcond=1e-10, square_root=False, check_stability=True):
    """ Pseudo-inverse of a symetric positive-definite matrix

    Parameters
    ----------
    a: array_like, shape (N, N),
       Symetric (not checked) positive-definite matrix to be pseudo-inverted.

    rcond: float, optional, default: 1e-10,
           Cutoff for small singular values.
           Singular values smaller (in modulus) than
           `rcond` * largest_singular_value (again, in modulus)
           are set to zero.

    square_root: boolean, optional, default: False,
                 return the matrix square-root of the pseudo-inverse instead

    Returns
    -------
    res: ndarray, shape (N, M)
         The pseudo-inverse of `a`
         or the (matrix) square-root of the pseudo-inverse.

    Raises
    ------
    IndefiniteError: if a is not positive-definite.

    Notes
    -----
    Uses the eigen-decomposition of `a`.

    Small modifications wrt numpy.linalg.pinv:
        - uses the eigen-decomposition instead of the svd
        - only the real part
        - check for positive-definiteness and eventually numerical stability
    """
    N, _N = a.shape
    assert N == _N, "Matrix is not square!"
    # get the eigen-decomposition
    # w, v = np.linalg.eigh(a)
    v, w, u = np.linalg.svd(a)
    sort_index = np.argsort(w)
    w = w[sort_index]
    v = v[:,sort_index]
    # check positive-definiteness
    ev_min = w.min()
    if ev_min <= 0:
        msg = "Matrix is not positive-definite: min ev = {0}"
        raise IndefiniteError(msg.format(ev_min))
    # check stability of eigen-decomposition
    if check_stability:
        # XXX use a preconditioner?
        if not np.allclose(a, np.dot(v, w[:, np.newaxis] * v.T)):
            raise NumericalError(
                "Instability in eigh (condition number={:g})".format(
                    (w.max() / w.min())))

    # invert the "large enough" part of s
    cutoff = rcond * w.max()
    for i in range(N):
        if w[i] > cutoff:
            if square_root:
                # square root of the pseudo-inverse
                w[i] = np.sqrt(1. / w[i])
            else:
                w[i] = 1. / w[i]
        else:
            w[i] = 0.
    # compute the pseudo-inverse (using broadcasting)
    res = np.real(np.dot(v, w[:, np.newaxis] * v.T))
    # check stability of pseudo-inverse
    if check_stability:
        if square_root:
            pa = np.dot(res, res)
            approx_a = np.dot(a, np.dot(pa, a))
            msg = "Instability in square-root of pseudo-inverse"
        else:
            approx_a = np.dot(a, np.dot(res, a))
            msg = "Instability in pseudo-inverse"
        if not np.allclose(a, approx_a):
            # be a bit laxist by looking at the Mean Squared Error
            mse = np.mean((a - approx_a) ** 2)
            if mse > 1e-16:
                raise NumericalError("{} (MSE={:g})".format(msg, mse))
    return res


class IndefiniteError(Exception):
    """Error raised on problematic non-positive-definiteness"""
    pass


class NumericalError(Exception):
    """Error raised on problems caused by numerical instability"""
    pass


def build_geom_neighbor_graph(geoms, n_neighbors):
    """ Computes the sparse CSR geometrical adjacency matrix gadj

    Parameters
    ----------
    geoms: (n_pts, d) array,
           the geometrical info

    n_neighbors: int,
                 number of neighbors

    Returns
    -------
    gadj: (n_pts, n_pts) sparse CSR array,
          the adjacency matrix
          gadj[i,j] == 1 iff i and j are geometrical neighbors

    Notes
    -----
    gadj might not be symmetric!
    """
    n_pts = geoms.shape[0]
    pyflann.set_distance_type('euclidean')  # squared euclidean actually
    fli = pyflann.FLANN()
    build_params = dict(algorithm='kdtree', num_neighbors=n_neighbors)
    gneighbs, _ = fli.nn(geoms, geoms, **build_params)
    data = np.ones((n_pts, n_neighbors), dtype='u1')
    indptr = np.arange(0, n_pts * n_neighbors + 1, n_neighbors, dtype=int)
    gadj = sparse.csr_matrix(
        (data.ravel(), gneighbs.ravel(), indptr), shape=(n_pts, n_pts))
    return gadj


def build_sym_geom_adjacency(geoms, max_gnn=100):
    """ Return the sparsest yet maximally connected symetric geometrical adjacency matrix
    """
    global INTERNAL_PARAMETERS
    min_gnn = INTERNAL_PARAMETERS['min_geom_neighbors']
    assert min_gnn < max_gnn, "Too high minimum number of neighbors"
    n_pts = geoms.shape[0]
    for n_neighbors in range(min_gnn, max_gnn + 1):
        # find the lowest number of NN s.t. the graph is not too disconnected
        C = build_geom_neighbor_graph(geoms, n_neighbors)
        neighbs = C.indices.reshape((n_pts, n_neighbors))
        C = C + C.T
        C.data[:] = 1
        n_comp, _ = sparse.cs_graph_components(C)
        if n_comp == 1:
            print "# use n_neighbors=%d" % n_neighbors
            break
        elif n_comp < 1:
            raise ValueError('Bug: n_comp=%d' % n_comp)
    if n_comp > 1:
        print "# use maximum n_neighbors=%d (%d components)" % (
            n_neighbors, n_comp)
    return n_comp, C, neighbs


class SplitError(Exception):
    pass


def allclose_rows(X):
    return np.sum(np.diff(X, axis=0) ** 2) < 1e-10


def get_kmeans_split(X):
    """ Returns the list of row labels obtained by k-means with k == 2
    """

    n_pts, n_dims = X.shape

    # special case: all rows are the same: k-means will hold forever...
    if allclose_rows(X):
        # all vectors are equal: cannot split
        sys.stderr.write('# WARNING: all rows are close\n')
        sys.stderr.flush()
        return None

    if n_pts > 1e3:
        model = MiniBatchKMeans(
            n_clusters=2, init="k-means++", max_iter=30, batch_size=1000,
            compute_labels=True, max_no_improvement=None, n_init=5)
    else:
        model = KMeans(n_clusters=2, init="k-means++", n_init=5, max_iter=100)

    model.fit(X)
    labels = model.labels_

    return labels


class PriorityQueue(object):
    """ Simple priority queue class on objects

    Compares objects based on their "minus_priority" property (must have this attribute)

    Implemented with a heap
    """

    def __init__(self):
        self._heap = []

    def __len__(self):
        return len(self._heap)

    def push(self, obj):
        """ Insert obj in the queue according to obj.minus_priority
        """
        # wrap the object to allow for correct pop operation
        # remember that in python it's a min-heap (not max!)
        wrap_obj = (obj.minus_priority, len(self), obj)
        # use insertion number to ensure we never compare based on obj itself!
        # additionally resolves ties by popping earliest-inserted object
        heapq.heappush(self._heap, wrap_obj)

    def pop(self):
        """ Returns the highest priority object in the queue

        Ties are resolved by popping the object inserted first (FIFO).
        """
        _, _, obj = heapq.heappop(self._heap)
        return obj


class SpectralNode(object):
    """ A node used to split points by thresholding a single eigen-vector

    Attributes
    ----------
    ids: (n, ) array,
         the (integer) indexes of points affected by this split

    vec: int,
         the eigen-vector number (dimension in the embedding) used for the split

    score: float,
           score of the node (reflects quality in terms of consistency and density)

    name: string,
          path of the node in string format
          (e.g. root is '1', left child of root is '10')

    has_children: boolean,
                  whether the node has children or not
                  (i.e. if it's a leaf or if it wasn't split yet)

    thresh: float,
            the threshold used to split along the projection on the selected eigen-vector
    """

    def __init__(self, ids, vec, score=None, name=""):
        """ A node corresponds to a split of points indexed by `ids`.
        """
        self.size = len(ids)
        self.ids = ids
        self.vec = vec
        self.score = 0. if score is None else score
        self.name = name  # binary string path: 0 for left, 1 for right
        self.has_children = False
        self.thresh = None

    @property
    def minus_priority(self):
        """ Defines lexical order on nodes used to decide splits

        In order of decreasing importance:
            1) nodes where the split is on smaller eigen-vectors (more reliable)
            2) nodes with the lowest parent score (highest gain expected),
            3) bigger nodes first

        Note that this property is the *opposite* of a priority
        """
        #return (-self.size, self.vec, self.score)  # kinda "depth-first"
        #return (self.vec, self.score, -self.size)  # kinda "breadth-first"
        return (self.score, -self.size, self.vec)  # kinda "depth-first with back-tracking"


def _ps(score):
    """ Convenience function for score printing
    """
    #s = "({0[0]:.3f}, {0[1]:.3f})".format(score)
    s = "{0:.3f}".format(score)
    return s


class SpectralTree(object):
    """ Binary tree used for hierarchical divisive clustering of a spectral embedding

    Attributes
    ----------
    labels: (n_pts, ) array,
            the cluster memberships according to the split up to the root (included)

    n_clusters: int,
                the number of clusters up to now

    Notes
    -----
    The tree is only implicit.
    """

    def __init__(self, E, ngeoms, min_leaf_size, max_leaf_size, min_leaves,
                 max_depth, min_evect_amplitude, split_type, n_threshs):
        """ Initialize with empty tree

        Parameters
        ----------
        E: (n_pts, n_vec) array,
           the spectral embedding of the points on n_vec eigen-vectors,

        ngeoms: (n_pts, 3) array,
                the spatio-temporal information of each point
                (assumed to be normalized)

        min_leaf_size: int,
                       the minimum size of a leaf
                       (don't split smaller nodes than this)

        max_leaf_size: int,
                       the maximum size of a leaf
                       (always split for nodes bigger than this)

        min_leaves: int,
                    minimum number of leaves for the full tree
                    (always split if less)

        max_depth: int,
                   don't split nodes deeper than this (< 63)

        min_evect_amplitude: float,
                             only used when thresholding to split
                             don't split a set of points along an eigenvector
                             with an amplitude (max-min) smaller than this
                             (e.g. 1e-10)

        split_type: str,
                    "threshold": threshold individual eigenvectors to split a node
                    "kmeans": use k-means to bi-partition a node

        n_threshs: int,
                   number of evenly-spaced in (0, 1) thresholds to try for splitting

        """

        self.E = E
        self.n_pts, self.n_vec = E.shape
        self.ngeoms = ngeoms
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.min_leaves = min_leaves
        self.max_depth = max(1, min(max_depth, 62))
        self.min_evect_amplitude = float(min_evect_amplitude)
        self.split_type = split_type
        self.n_threshs = n_threshs

        # checks
        assert self.n_pts == self.ngeoms.shape[0], "Invalid geoms dimension"
        assert self.split_type in ("threshold", "kmeans"), "Unknown split_type"
        assert self.max_leaf_size >= self.min_leaf_size, "max_leaf_size < min_leaf_size"
        assert min_evect_amplitude > 0, \
            "min_evect_amplitude == {} <= 0".format(min_evect_amplitude)

        # split-type specific treatments
        if self.split_type == "kmeans":
            # l2-normalize E
            nrlz = np.sqrt((self.E ** 2).sum(axis=1))
            mask = nrlz > 0
            self.E[mask] /= nrlz[mask][:, np.newaxis]
        elif self.split_type == "threshold":
            # rescale projections to be between 0 and 1
            self.E -= self.E.min(axis=0)[np.newaxis, :]
            nrlz = self.E.max(axis=0)
            mask = nrlz != 0
            self.E[:, mask] /= nrlz[mask][np.newaxis, :]
            # relative per-dim thresholds (min 10% - 90% split imbalance)
            self.percentiles = np.linspace(0.10, 0.90, num=self.n_threshs)

        # build the geom adjacency matrix (used for scoring)
        _, self._gadj, self._gneighbs = build_sym_geom_adjacency(ngeoms)

    def _get_tube_connectedness(self, tube_idxs):
        """ Return the connectedness measure of the tube

        Parameters
        ----------
        tube_idxs: (tube_size, ) array,
                   the ids of the points in the tube we're interested in

        Returns
        -------
        connectedness: float in [0, 1],
                       1/#connected components
        """
        # extract the rows of self._gadj which are in the tube
        ids = self._gadj.indices
        iptr = self._gadj.indptr
        sub_indices = np.hstack(
            [ids[iptr[i]:iptr[i + 1]] for i in tube_idxs]).astype(ids.dtype)
        sub_indptr = np.zeros_like(iptr)
        sub_indptr[tube_idxs + 1] = iptr[tube_idxs + 1] - iptr[tube_idxs]
        sub_indptr = np.cumsum(sub_indptr, dtype=iptr.dtype)
        _conn_labs = np.empty((self.n_pts,), dtype=iptr.dtype)
        num_conn = cs_graph_components(
            self.n_pts, sub_indptr, sub_indices, _conn_labs)
        assert num_conn > 0, "BUG: negative or null num_conn %d" % num_conn
        connectedness = 1. / num_conn
        return connectedness

    def _get_tube_label_density(self, tube_idxs):
        """ Return the average local label agreement of the tube

        Parameters
        ----------
        tube_idxs: (tube_size, ) array,
                   the ids of the points in the tube we're interested in

        Returns
        -------
        density: float in [0, 1],
                 average ratio of geometrical neighbors in the tube
        """
        # get the indexes of the nearest neighbors of all tube points
        gneighbs = self._gneighbs[tube_idxs]
        # count the number of neighbors in the tube
        fbl = np.zeros((self.n_pts, ), dtype=bool)
        fbl[tube_idxs] = True
        nnt = fbl[gneighbs].sum()
        assert nnt > len(
            tube_idxs), "BUG: at least the points are in the tube!"
        # get the overall ratio
        density = float(nnt) / (gneighbs.shape[0] * gneighbs.shape[1])
        return density

    # XXX use numexpr and (x-y)**2 instead?
    def _get_tube_inertia(self, tube_idxs):
        """ Return the within-cluster variance (like in k-means)

        Parameters
        ----------
        tube_idxs: (tube_size, ) array,
                   the ids of the points in the tube we're interested in

        Returns
        -------
        inertia: float,
                 the sum of square differences from the mean
        """
        # get the features of the in-cluster points
        X = self.E[tube_idxs]
        # get the centroid
        centroid = np.mean(X, axis=0)
        # compute the sum of the squared norms
        inertia = np.sum(X * X)
        inertia += len(tube_idxs) * np.sum(centroid * centroid)
        # compute the inner-products with the centroid
        inertia -= 2 * np.sum(np.dot(X, centroid))
        return inertia

    # XXX critical part: find good scoring!
    def get_tube_score(self, tube_idxs):
        """ Return the score of a single cluster

        Parameters
        ----------
        tube_idxs: (tube_size, ) array,
                   the ids of the points in the tube we're interested in

        Returns
        -------
        score: float,
               the quality score (the higher the better) of the cluster
               we use as score, the inverse of the number of connected components

        """
        assert len(
            tube_idxs) > 0, "BUG: #tube_idxs == {0}".format(len(tube_idxs))
        # get the connectedness
        tc = np.sqrt(self._get_tube_connectedness(tube_idxs))
        return tc

    def _get_candidate_thresholds(self, node, vec):
        """ Return a list of pairs (n_vec, thresh) of a threshold applicable to
        the n_vec'th dimension of the spectral embedding (eigenvector n_vec)
        """

        if vec >= self.n_vec:
            msg = "BUG: try to split on {0} which is after max_n_vec ({1})"
            raise SplitError(msg.format(vec, self.n_vec))

        # the projections on the selected eigen-vector
        evs = self.E[node.ids, vec]

        # get the thresholds
        _scale = evs.max() - evs.min()
        if _scale < self.min_evect_amplitude:
            # not enough amplitude to split
            used_threshs = []
        else:
            # get quantiles as thresholds
            evs.sort()
            _threshs = evs[(self.percentiles * (len(evs) - 1)).astype(int)]
            # discard thresholds very close to each other
            # (unstable: small change yields very different split)
            used_threshs = [_threshs[0]]  # always use the first one
            for _t in _threshs[1:]:
                if (_t - used_threshs[-1]) > 1e-2 * _scale:
                    # keep: gap between thresholds is more than 1% of total scale
                    used_threshs.append(_t)

        if len(used_threshs) == 0:
            msg = "WARNING: too small amplitude ({0:0.1e})"
            msg += " or too close thresholds to split node {1} at vec {2}\n"
            sys.stderr.write(msg.format(_scale, node.name, vec))
            sys.stderr.flush()

        return used_threshs

    def _split_threshold(self, node):
        """Find the best split of a node by thresholding the corresponding eigen-vector
        """

        # define the score to improve upon
        if self.n_clusters >= self.min_leaves and node.size <= self.max_leaf_size:
            # split only if min(children scores) > node.score
            force_split = False
            best_score = node.score
        else:
            # force split: just take the best (even if children are worse)
            force_split = True
            best_score = None

        left, right = None, None

        # iterate over embedding dimensions (first ones are more reliable)
        # up to max_n_vec (included), until we found an improving split
        for _vec in range(self.n_vec):

            # get the candidate thresholds along this dimension
            threshs = self._get_candidate_thresholds(node, _vec)

            # look for an improving best split along this eigenvector
            for _t in threshs:
                # compute the split
                below_thresh = self.E[node.ids, _vec] < _t
                _lids = node.ids[below_thresh]
                _rids = node.ids[np.logical_not(below_thresh)]
                # check if the tubes are not too small
                _nl, _nr = len(_lids), len(_rids)
                is_valid = _nl >= self.min_leaf_size and _nr >= self.min_leaf_size
                if is_valid:
                    # compute the score of the new tubes only
                    _sl = self.get_tube_score(_lids)
                    _sr = self.get_tube_score(_rids)
                    # get the score of this split
                    split_score = min(_sl, _sr)
                    if best_score is None or split_score > best_score:
                        # better split
                        best_score = split_score
                        node.has_children = True
                        node.thresh = _t
                        left = SpectralNode(
                            _lids, _vec, score=_sl, name=node.name + "0")
                        right = SpectralNode(
                            _rids, _vec, score=_sr, name=node.name + "1")

            # check stopping criterion
            if node.has_children:
                # we found an improving split
                if _vec > 0 or not force_split:
                    # found an improving non-forced split: stop here
                    break

        return left, right

    def _split_kmeans(self, node):
        """Find the best split of a node by using k-means with k=2 on the full embedding
        """

        # bi-partition with k-means until children have enough samples or max outliers is reached
        n_outliers = 0
        ids = node.ids
        left, right = None, None

        # define the score to improve upon
        if self.n_clusters >= self.min_leaves and node.size <= self.max_leaf_size:
            # require an improvement of children
            best_score = node.score
            # limit outliers to smallest cluster possible
            max_outliers = self.min_leaf_size
        else:
            # just take the best split (even if children are worse)
            best_score = None
            # no limit on outliers: always split
            max_outliers = np.inf

        # iterate until valid split or reached max outliers
        while n_outliers < max_outliers:
            labels = get_kmeans_split(self.E[ids])
            if labels is None:
                # could not split
                break
            # compute the split
            _lids = ids[labels == 0]
            _rids = ids[labels == 1]
            # check if the tubes are not too small
            _nl, _nr = len(_lids), len(_rids)
            if _nl + _nr != len(ids):
                raise SplitError("BUG in kmeans")
            if _nl >= self.min_leaf_size and _nr >= self.min_leaf_size:
                # both children are large enough
                _sl = self.get_tube_score(_lids)
                _sr = self.get_tube_score(_rids)
                # get the score of this split
                score = min(_sl, _sr)
                # check if the split improves (each child has better score than the parent)
                if best_score is None or score > best_score:
                    # register the split (vec is used to store depth in the tree)
                    node.has_children = True
                    best_score = score
                    left = SpectralNode(
                        _lids, node.vec + 1, score=_sl, name=node.name + "0")
                    right = SpectralNode(
                        _rids, node.vec + 1, score=_sr, name=node.name + "1")
                break
            elif _nl < self.min_leaf_size and _nr >= self.min_leaf_size:
                # left children is too small: add as outlier
                self.labels[_lids] = -1
                n_outliers += _nl
                # carry on with this subset
                ids = _rids
            elif _nr < self.min_leaf_size and _nl >= self.min_leaf_size:
                # right children is too small: add as outlier
                self.labels[_rids] = -1
                n_outliers += _nr
                # carry on with this subset
                ids = _lids
            else:
                # both too small: node is a leaf
                #msg = 'Both children are too small:'
                #msg+= ' too many outliers ({0} >= max_outliers={1})'.format(n_outliers, max_outliers)
                #msg+= ' or too small node size ({0})'.format(node.size)
                #raise SplitError(msg)
                break

        return left, right

    def _split_forced(self, node):
        """Force the split of a node, disregarding node size constraints

        The split is not random but is obtained by cutting in 2 equally-sized
        children sorted according of the projection along the first eigenvector.

        The use of this function is only as a last resort to force a mandatory
        split if normal splitting strategies have failed.
        """
        # compute the split
        _vec = 0
        sorted_idxs = np.argsort(self.E[node.ids, _vec]).squeeze()
        n = len(sorted_idxs) // 2
        _lids = node.ids[sorted_idxs[:n]]
        _rids = node.ids[sorted_idxs[n:]]
        # compute the score of the new tubes only
        _sl = self.get_tube_score(_lids)
        _sr = self.get_tube_score(_rids)
        # register the split
        node.has_children = True
        node.thresh = np.median(self.E[node.ids, _vec])  # arbitrary
        # Note: median would not ensure equal size (because of duplicate values)
        left = SpectralNode(_lids, _vec, score=_sl, name=node.name + "0")
        right = SpectralNode(_rids, _vec, score=_sr, name=node.name + "1")

        return left, right

    def split(self, node):
        """Split a tree in two

        Parameters
        ----------
        node: SpectralNode object,
              the node of the subtree we want to split
              (contains the eigen-vector along which we split)

        Returns
        -------
        left: SpectralNode object,
              the root of the left subtree (None for leaves)

        right: SpectralNode object,
               the root of the right subtree (None for leaves)

        Notes
        -----
        Additionally updates the labels and number of clusters.
        """
        # check node was not already split
        if node.has_children:
            raise SplitError("BUG: node was already split")

        # early stopping (only if enough nodes already)
        if self.n_clusters >= self.min_leaves:
            # make a leaf if too small to split
            if node.size <= 2 * self.min_leaf_size:
                return None, None
            # special case: make a leaf if too deep already
            if len(node.name) > self.max_depth:
                # int(node.name, 2) is too big to be represented as a long (int64)
                # if len(node.name > 62)
                sys.stderr.write('# WARNING: early stopping too deep branch'
                                 ' {}\n'.format(node.name))
                sys.stderr.flush()
                return None, None

        # bi-partition the node's samples
        if self.split_type == "kmeans":
            left, right = self._split_kmeans(node)
        else:
            left, right = self._split_threshold(node)

        # check if we have two leaves or none
        if (left is None and right is not None) or (left is not None and right is None):
            raise SplitError(
                "BUG: both children should be simultaneously"
                "either None or not")

        # check the post-conditions
        if left is None or right is None:
            # node is a leaf
            if node.has_children:
                raise SplitError("BUG: leaf node marked with (empty) children")
            # check if it must have been split instead of being a leaf
            if node.size > self.max_leaf_size:
                # force the split
                left, right = self._split_forced(node)
                msg = 'WARNING: forced to split a must-split node that was'
                msg += ' too big to be a leaf ({0} > max_leaf_size={1})\n'
                sys.stderr.write(msg.format(node.size, self.max_leaf_size))
            if self.n_clusters < self.min_leaves:
                # force the split
                left, right = self._split_forced(node)
                msg = 'WARNING: forced to split a must-split node that had'
                msg += ' not enough clusters ({0} < min_leaves={1})\n'
                sys.stderr.write(msg.format(self.n_clusters, self.min_leaves))

        # finalize the split
        if node.has_children:
            # update the labels of right child only (left keeps the same)
            self.labels[right.ids] = self.n_clusters
            self.n_clusters += 1

        return left, right

    def build(self, verbose=True):
        """Recursively split in two, starting from a cluster containing all points

        The nodes to split are decided based on a priority queue (cf. SpectralNode).
        """
        # initially: one cluster
        self.labels = np.zeros((self.n_pts, ), dtype=int)
        self.int_paths = np.zeros((self.n_pts, ), dtype=int)
        self.n_clusters = 1

        # create the root and add it to a FIFO queue of nodes to process
        root = SpectralNode(
            np.arange(self.n_pts), 0, name="1")  # '1' by convention
        to_split = PriorityQueue()
        to_split.push(root)

        # recursively split
        #nrecs = 0
        while len(to_split) > 0:
            # get the node with highest priority
            node = to_split.pop()
            left, right = self.split(node)

            # push to the priority queue
            if node.has_children:
                # node was split: push the children
                to_split.push(left)
                to_split.push(right)
            else:
                # node is a leaf: update the cluster tree paths for the concerned points
                self.int_paths[node.ids] = int(node.name, 2)
                # Note: outliers (not in node.ids) have default '0' path

            # to save all partial labelings, do
            #nrecs += 1
            #np.save('labels_%04d_split_%s.npy' % (nrecs, node.name), self.labels)

            if verbose:
                self._print_split_infos(node, left, right, len(to_split))

        # check we don't have a too small number of leaves
        assert self.n_clusters >= self.min_leaves, \
            "BUG: not enough clusters {0}".format(self.n_clusters)

    def _print_split_infos(self, node, left, right, left_to_split):
        """ Print DEBUG infos about the split of 'node' in 'left' and 'right'
        """
        DEBUG_info = "#DEBUG n_clusters={n_clusters:04d} to_split={to_split:04d}"
        infos = dict(n_clusters=self.n_clusters, to_split=left_to_split)
        DEBUG_info += " score={score}"
        infos['score'] = _ps(node.score)
        if node.has_children:
            # node was split
            DEBUG_info += " vec={vec:04d} sl={sl} nl={nl:06d} sr={sr} nr={nr:06d}"
            infos['vec'] = left.vec
            infos['sl'] = _ps(left.score)
            infos['nl'] = left.size
            infos['sr'] = _ps(right.score)
            infos['nr'] = right.size
        else:
            # node is a leaf
            DEBUG_info += " LEAF" + ' ' * 42
        DEBUG_info += " size={size:06d} path={path}"
        infos['size'] = node.size
        infos['path'] = node.name
        print DEBUG_info.format(**infos)
        sys.stdout.flush()