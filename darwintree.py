__author__ = 'aclapes'

from os.path import isfile, isdir, exists, join, splitext, basename, dirname
from os import makedirs
import cPickle
import time
import numpy as np

from videodarwin import darwin

INTERNAL_PARAMETERS = dict(

)

def darwin(fullfeatnames, st, num_videos, darwins_path):
    if not exists(darwins_path):
        makedirs(darwins_path)

    for feat_t in fullfeatnames:
        # node_darwins[feat_t] = dict()

        if not exists(join(darwins_path, feat_t)):
            makedirs(join(darwins_path, feat_t))

        for featname in fullfeatnames[feat_t]:
            output_filepath = join(darwins_path, feat_t, basename(featname))
            if isfile(output_filepath):
                print('%s -> OK' % (featname))
                continue

            start_time = time.time()

            with open(featname, 'rb') as f:
                data = cPickle.load(f)

            # compute VD
            node_darwins = dict()
            node_darwins[1] = darwin(data['X'])
            for id, X in data['tree_perframe'].iteritems():
                node_darwins[id] = darwin(X)

            # construct a list of edge pairs for easy access

            with open(output_filepath, 'wb') as f:
                cPickle.dump(dict(node_darwins=node_darwins), f)

            elapsed_time = time.time() - start_time
            print('%s -> DONE (in %.2f secs)' % (output_filepath, elapsed_time))

    return None


# ==============================================================================
# Helper functions
# ==============================================================================