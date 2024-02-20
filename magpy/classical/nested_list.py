"""
Provides a data format to represent nested lists of numerical values in a way
that is suitable for numba jit optimization.
"""

import numpy as np
from numba import njit


def new(list_, depth):
    if depth <= 0:
        return list_, [[list_.__len__()]]   # need to use __len__() because we overrode len() below
    
    list_flattened = []
    separator_multiidxs = []
    current_flat_idx = 0
    for idx, sublist in enumerate(list_):
        sublist_flattened, sublist_separator_multiidxs = new(sublist, depth-1)
        list_flattened += [*sublist_flattened]
        separator_multiidxs += [
            [
                idx, 
                *sublist_separator_multiidx[:-1], 
                sublist_separator_multiidx[-1] + current_flat_idx
            ] for sublist_separator_multiidx in sublist_separator_multiidxs
        ]
        current_flat_idx += sublist_flattened.__len__()

    return np.array(list_flattened), np.array(separator_multiidxs)




@njit
def get(list_, idx):
    flat_list, separator_multiidxs = list_
    pos_of_multiidxs_for_prev_idx = np.where(separator_multiidxs[:, 0] == idx-1)
    if pos_of_multiidxs_for_prev_idx[0].shape[0] > 0:
        pos_of_last_multiidx_for_prev_idx = pos_of_multiidxs_for_prev_idx[-1][-1]
        start_idx_flat = separator_multiidxs[pos_of_last_multiidx_for_prev_idx, -1]
    else:
        start_idx_flat = 0
        
    pos_of_multiidxs_for_idx = np.where(separator_multiidxs[:, 0] == idx)
    pos_of_last_multiidx_for_idx = pos_of_multiidxs_for_idx[-1][-1]
    end_idx_flat = separator_multiidxs[pos_of_last_multiidx_for_idx, -1]

    sublist = flat_list[start_idx_flat:end_idx_flat]
    sublist_separator_multiidxs = separator_multiidxs[pos_of_multiidxs_for_idx][:, 1:]
    sublist_separator_multiidxs[:, -1] -= start_idx_flat

    return sublist, sublist_separator_multiidxs


@njit
def len(list_):
    _, separator_multiidxs = list_
    return np.amax(separator_multiidxs[:, 0]) + 1