import numpy as np
from numba import njit




#@njit
def convert_to_flat_index(bravais_coords, subl_idx,
                          lattice_sizes, num_sites_unit_cell):
    flat_idx = subl_idx
    factor = num_sites_unit_cell
    bravais_coords = bravais_coords
    for i in range(len(bravais_coords)-1, -1, -1):
        flat_idx += factor * bravais_coords[i]
        factor *= lattice_sizes[i]
    return flat_idx



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