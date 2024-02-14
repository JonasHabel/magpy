import numpy as np
from numba import njit




@njit
def convert_to_flat_index(bravais_coords, subl_idx,
                          lattice_sizes, num_sites_unit_cell):
    flat_idx = subl_idx
    factor = num_sites_unit_cell
    for i in range(len(bravais_coords)-1, -1, -1):
        flat_idx += factor * bravais_coords[i]
        factor *= lattice_sizes[i]
    return flat_idx



"""
--- jit boilerplate implementation ---

Takes a tensor A of shape (n, n, ...(repeat N times)..., n) and an array of
vectors b of shape (N, n) and performs the tensor contraction
    A_{ijk...} * b_{1i} * b_{2j} * b_{3k} ...
"""
@njit
def tensor_contract(A, b):
    if len(A.shape) <= 0:
        return A
    
    partial_contraction = np.tensordot(b[0], A, axes=[[0], [0]])

    return tensor_contract(partial_contraction, b[1:])