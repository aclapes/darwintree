import numpy as np
import copy
import xmltodict

# ==============================================================================
# Configuration functions
# ==============================================================================

def load_XML_config(filepath):
    """
    Read the configuration from a .xml file in disk.
    This sets some paths and the features to use.
    :param filepath:
    :return:
    """
    config_dict = dict()

    with open(filepath) as fd:
        xml = xmltodict.parse(fd.read())

    for path in xml['configuration']['path']:
        config_dict[path['@key']] = path['#text'].encode('utf-8')

    if 'option' in xml['configuration']:
        if not isinstance(xml['configuration']['option'], list):
            option = xml['configuration']['option']
            config_dict[option['@key']] = option['#text'].encode('utf-8')
        else:
            for option in xml['configuration']['option']:
                config_dict[option['@key']] = option['#text'].encode('utf-8')
        # data type conversions from str to target type
        if 'num_threads' in config_dict:  # num threads has to be an integer
            config_dict['num_threads'] = int(config_dict['num_threads'])

    features_list = xml['configuration']['features_list']

    if type(features_list['item']) is unicode:
        config_dict.setdefault('features_list',[]).append(features_list['item'].encode('utf-8'))
    else:
        for item in features_list['item']:
            feat = item.encode('utf-8')
            config_dict.setdefault('features_list',[]).append(feat)

    # methods_list = xml['configuration']['methods_list']
    # if type(methods_list['item']) is unicode:
    #     config_dict.setdefault('methods_list',[]).append(methods_list['item'].encode('utf-8'))
    # else:
    #     for item in methods_list['item']:
    #         method = item.encode('utf-8')
    #         config_dict.setdefault('methods_list',[]).append(method)

    return config_dict


def get_global_config(xml_config):
    """
    Construct some additional (output) paths from the xml configuration.
    :param xml_config:
    :return:
    """
    parent_path = xml_config['data_path'] + xml_config['dataset_name'] + '/'
    if not isdir(parent_path):
        makedirs(parent_path)

    tracklets_path = parent_path + 'tracklets/'
    clusters_path = parent_path + 'clusters/'
    intermediates_path = parent_path + 'intermediates/'
    feats_path = parent_path + 'feats/'
    kernels_path = parent_path + 'kernels/'

    return tracklets_path, clusters_path, intermediates_path, feats_path, kernels_path


# ==============================================================================
# Data structures handling functions
# ==============================================================================

def merge_dictionaries(dicts):
    """
    Merges all the dictionaries in dicts in one and only dictionary.
    This function uses internally merge_pair_of_dictionaries. Please refer to
    its documentation to know more about the merging.
    :param dicts:
    :return: merge_dict
    """
    merge_dict = copy.deepcopy(dicts[0])
    for dict in dicts[1:]:
        merge_dict = merge_pair_of_dictionaries(merge_dict, dict)

    return merge_dict


def merge_pair_of_dictionaries(dst, src):
    """
    Merges two dictionaries recursively, being a deep version of the update function from python dicts.
    It is based on the code provided in:
    http://code.activestate.com/recipes/499335-recursively-update-a-dictionary-without-hitting-py/
    This version has been modified so as to do not overwrite "dst" when there' a hit, but keep both ("src" and "dst")
    in a list.
    :param dst:
    :param src:
    :return: recursively updated "dst"
    """
    stack = [(dst, src)]
    while stack:
        current_dst, current_src = stack.pop()
        for key in current_src:
            if key not in current_dst:
                current_dst[key] = current_src[key]
            else:
                if isinstance(current_src[key], dict) and isinstance(current_dst[key], dict) :
                    stack.append((current_dst[key], current_src[key]))
                else:
                    # here it is my modification (aclapes)
                    if not isinstance(current_dst[key], list):
                        current_dst[key] = [current_dst[key], current_src[key]]  # there is hit
                    else:
                        current_dst[key].append( (current_src[key]) if not isinstance(current_src[key],list) else current_src[key] )  # create a list and keep both, do not overwrite
    return dst


def serialize_nested_dictionary(nested_dict):
    stack = [nested_dict]
    serialized_dict = []
    while stack:
        current_dict = stack.pop()
        for key in current_dict:
            if isinstance(current_dict[key], dict):
                stack.append(current_dict[key])
            else:
                serialized_dict += current_dict[key]

    return serialized_dict


def sum_of_arrays(arrays, weights=None, norm=None):
    if weights is None:
        weights = [1.0/len(arrays)] * len(arrays)
    if norm is 'median':
        A, _ = normalize(arrays[0], norm)
    if norm is 'sqrt':
        A = np.sqrt(arrays[0])

    if isinstance(arrays, np.ndarray):
        S = weights[0] * (arrays if norm is None else A)
    else:
        S = weights[0] * (arrays[0] if norm is None else A) # element-wise sumation of arrays
        if isinstance(arrays, list):
            for i in range(1,len(arrays)):
                if norm is 'median':
                    A, _ = normalize(arrays[i], norm)
                elif norm is 'sqrt':
                    A = np.sqrt(arrays[i])

                S += weights[i] * (arrays[i] if norm is None else A)  # accumulate next array

    return S

# def normalize_by_median(K, p=None):
#     if p is None:
#         values = K[K != 0]
#         p = 1.0 / np.nanmedian(values) if values != [] else 1.0
#     return p*K, p

# def normalization(K,type='median'):
#     p = 1.0
#
#     values = K[K != 0]
#     if values != []:
#         if type == 'mean':
#             p = 1.0/np.nanmean(values)
#         elif type == 'median':
#             p = 1.0/np.nanmedian(values)
#
#     return K, p

def normalize(K, type='mean'):
    p = argnormalize(K, type=type)
    return p * K

def normalization(K, type='mean'):
    p = argnormalize(K, type=type)
    return p * K, p

def argnormalize(K,type='mean'):
    p = 1.0

    values = K[K != 0]
    if values != []:
        if type == 'mean':
            p = 1.0/np.nanmean(values)
        elif type == 'median':
            p = 1.0/np.nanmedian(values)

    return p

def uniform_weights_dist(n_weights, step=0.1):
    D = []
    w = n_weights * [0.]
    pos = 0
    acc_weight = 0.

    stack = []
    stack.append((w,pos,acc_weight))

    while len(stack) > 0:
        w, pos, acc_weight = stack.pop()

        if pos >= len(w)-1 or acc_weight == 1.:
            w[-1] = max(0, 1. - min(acc_weight,1))
            D.append(w[:])
            continue

        for x in np.arange(0, 1.-acc_weight+step, step):
            ww = w[:]
            ww[pos] = x
            stack.append((ww,pos+1,acc_weight+x))

    return D

def posneg(X, axis=0, copy=True):
    # get the negative part
    X_neg = np.zeros_like(X)
    X_neg[X < 0] = -X[X < 0]
    # get the positive part
    X_pos = None
    if not copy:
        X[X < 0] = 0
        X_pos = X
    else:
        X_pos = X.copy()
        X_pos[X < 0] = 0

    return np.concatenate([X_pos,X_neg], axis=(0 if axis == 1 else 1))

def rootSIFT(X):
    '''
    :param X: rootSIFT operation applied to elements of X (element-wise).
    Check Fisher Vectors literature.
    :return:
    '''
    return np.multiply(np.sign(X), np.sqrt(np.abs(X)))

