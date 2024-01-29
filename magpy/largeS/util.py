import itertools
import numpy as np
from magpy.largeS import real_space


def get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order):
    return interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_magnon_Hamiltonian(model, order)


def iterator(k_arrays, dim, func):
    num_ks = np.array([len(k_array) for k_array in k_arrays], dtype=np.int64)
    partial_modulos = np.array([
        np.prod(num_ks[n+1:]) for n in range(len(num_ks))
    ])

    for i in range(np.prod(num_ks)):
        k_multiidx = convert_1d_index_into_multiindex(i, partial_modulos)
        ks = np.array([
            k_array[k_multiidx[n]] for n, k_array in enumerate(k_arrays)
        ])
        if len(k_arrays) == 0:
            ks = ks.reshape((0, dim))
        yield k_multiidx, func(k_multiidx, ks)


def convert_1d_index_into_multiindex(idx, partial_modulos):
    quotient, remainder = (0, idx)
    multiidx = np.zeros(len(partial_modulos), dtype=np.int64)
    for n, N in enumerate(partial_modulos):
        quotient, remainder = remainder // N, remainder % N
        multiidx[n] = quotient
        
    return multiidx


def get_permutations(num_elements):
    return list(itertools.permutations(range(num_elements)))


def invert_permutation(permutation):
    permutation_np = np.array(permutation)
    inv = np.empty_like(permutation_np)
    inv[permutation_np] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def permute(arr, perm):
    return np.array([arr[perm[i]] for i in range(len(arr))])