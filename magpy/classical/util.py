import numpy as np
from numba import njit




@njit
def convert_to_flat_index(bravais_coords, subl_idx,
                          lattice_sizes, num_sublattices):
    flat_idx = int(subl_idx)
    factor = num_sublattices
    bravais_coords = bravais_coords
    
    for i in range(len(bravais_coords)-1, -1, -1):
        flat_idx += factor * bravais_coords[i]
        factor *= lattice_sizes[i]
        
    return int(flat_idx)



"""
--- jit boilerplate implementation ---

Takes a tensor A of shape (n, n, ...(repeat N times)..., n) and an array of
vectors b of shape (M<N, n) and performs the tensor contraction
    result_{i_1, ..., i_{N-M}} = A_{i_1,...,i_N} * b_{1,i_{N-M+1}} * b_{2,i_{M+1}} ... * b_{M,i_N}
using recursion of depth M
"""
#@njit
def tensor_contract(A, b):
    if len(b) <= 0:
        return A
    
    N = int(np.prod(np.array(A.shape[:-1])))
    A_flat = A.reshape((N, A.shape[-1]))
    partial_contraction_flat = A_flat @ b[-1]
    partial_contraction = partial_contraction_flat.reshape(A.shape[1:])

    return tensor_contract(partial_contraction, b[:-1])


@njit
def tensor_contract_jit(A, b, first_arg_is_flat=False):
    A_shape = len(b[0]) * np.ones(int(len(A) ** (1/len(b[0]))), dtype=np.int64) \
        if first_arg_is_flat else np.array(A.shape)
    num_idxs = len(A_shape)
    num_contracted_idxs = len(b)
    num_uncontracted_idxs = num_idxs - num_contracted_idxs
    contracted_shape = A_shape[num_uncontracted_idxs:]
    uncontracted_shape = A_shape[:num_uncontracted_idxs]
    contracted_shape_flat = int(np.prod(contracted_shape))
    uncontracted_shape_flat = int(np.prod(uncontracted_shape))
    partial_modulos = np.array([
        int(np.prod(A_shape[num_uncontracted_idxs:-n-1])) \
        for n in range(num_contracted_idxs)
    ])

    A_flat = A.reshape((uncontracted_shape_flat, contracted_shape_flat))
    result_flat = np.zeros(uncontracted_shape_flat, dtype=A.dtype)

    for i in range(uncontracted_shape_flat):
        for flat_idx in range(contracted_shape_flat):
            result_at_flat_idx = A_flat[i, flat_idx]
            b_idx = flat_idx
            for n in range(len(b)):
                quotient = b_idx // partial_modulos[n]
                remainder = b_idx % partial_modulos[n]
                result_at_flat_idx *= b[n, quotient]
                b_idx = remainder
            result_flat[i] += result_at_flat_idx

    return result_flat





"""
deep-flatten a nested list of numpy arrays into a 1d list
"""
def deep_flatten_to_1d_list(list_, depth):
    if depth <= 0:
        return list_, [[len(list_)]]
    
    list_flattened = []
    separator_multiidxs = []
    current_flat_idx = 0
    for idx, sublist in enumerate(list_):
        sublist_flattened, sublist_separator_multiidxs = \
            deep_flatten_to_1d_list(sublist, depth-1)
        list_flattened += [*sublist_flattened]
        separator_multiidxs += [
            [
                idx, 
                *sublist_separator_multiidx[:-1], 
                sublist_separator_multiidx[-1] + current_flat_idx
            ] for sublist_separator_multiidx in sublist_separator_multiidxs
        ]
        current_flat_idx += len(sublist_flattened)

    return list_flattened, separator_multiidxs


def deep_flatten_to_1d_array(list_, depth):
    list_flattened, separator_multiidxs = deep_flatten_to_1d_list(list_, depth)
    return np.array(list_flattened), np.array(separator_multiidxs)

@njit
def get_from_flat(flat_list, separator_multiidxs, idx):
    pos_of_multiidxs_for_prev_idx = np.where(separator_multiidxs[:, 0] == idx-1)
    if len(pos_of_multiidxs_for_prev_idx[0]) > 0:
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
def len_flat(flat_arr, separator_multiidxs):
    return np.amax(separator_multiidxs[:, 0]) + 1
