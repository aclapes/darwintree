import numpy as np
import copy

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
                        current_dst[key] += [current_src[key]] if not isinstance(current_dst[key],list) else current_src[key]  # create a list and keep both, do not overwrite
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


def sum_of_arrays(arrays, weights=None, norm=None, gamma=None):
    if weights is None:
        weights = [1.0/len(arrays)] * len(arrays)
    if norm is 'median':
        A, _ = normalize_by_median(arrays[0])
    if norm is 'sqrt':
        A = np.sqrt(arrays[0])

    S = weights[0] * (arrays[0] if norm is None else A) # element-wise sumation of arrays
    for i in range(1,len(arrays)):
        if norm is 'median':
            A, _ = normalize_by_median(arrays[i])
        elif norm is 'sqrt':
            A = np.sqrt(arrays[i])

        S += weights[i] * (arrays[i] if norm is None else A)  # accumulate next array

    return S

def normalize_by_median(K, p=None):
    if p is None:
        values = K[K != 0]
        p = 1.0 / np.median(values) if values != [] else 1
    return p*K, p