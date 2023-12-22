import numpy as np
from operator import itemgetter
from .lattice import ReciprocalLattice





def tensor_rotate(tensor, rot_matrices):
    if len(tensor.shape) == 1:  # tensor is a vector
        return np.tensordot(rot_matrices[0], tensor, axes=[[0], [0]])
    
    rotated_tensor = np.zeros(tensor.shape, dtype=complex)
    # apply rotation to axes 1, ..., len(tensor.shape)-1
    for i in range(tensor.shape[0]):
        rotated_tensor[i] = tensor_rotate(tensor[i], rot_matrices[1:])

    # apply rotation to axis 0
    rotated_tensor = np.tensordot(rot_matrices[0], rotated_tensor,
        axes=[[0], [0]])
    
    return rotated_tensor


__ID3 = np.identity(3)
LEVI_CIVITA = np.array([[[
        np.cross(__ID3[i], __ID3[j]).dot(__ID3[k]) for k in range(3)]
    for j in range(3)]
for i in range(3)])

def BOGO_METRIC(num_bands):
    return np.diag([1, -1] * num_bands)

PAULI_MATRICES = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])

LARGE_S_EXPANSION_COEFF = [np.sqrt(2), -1/np.sqrt(8)]


def permute(arr, perm):
    return [arr[perm[i]] for i in range(len(arr))]


def generate_einsum_indices(arr):
    return [chr(97 + element) for element in arr]
