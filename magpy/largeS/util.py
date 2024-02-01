import itertools
import numpy as np
from magpy.largeS import real_space


def get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order):
    return interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_magnon_Hamiltonian(model, order)


def flat_iterator(quantity_arrs, default_shape, func, iteration_dim=0):
    if len(quantity_arrs) >= 2:
        for quantity_arr in quantity_arrs:
            assert quantity_arr.shape[:iteration_dim] == quantity_arrs[0].shape[:iteration_dim]
            assert quantity_arr.shape[iteration_dim+1:] == quantity_arrs[0].shape[iteration_dim+1:]

    dimensions = np.array(
        [quantity_arr.shape[iteration_dim] for quantity_arr in quantity_arrs], 
        dtype=np.int64
    )

    def wrapped_func(multiidx, flat_idx):
        quantities_at_idx = np.array([
            quantity_arr[(*((slice(None),)*(iteration_dim-1)), multiidx[n])] \
            for n, quantity_arr in enumerate(quantity_arrs)
        ])
        if len(quantity_arrs) == 0:
            quantities_at_idx = quantities_at_idx.reshape((0, *default_shape))
        return func(multiidx, flat_idx, quantities_at_idx)

    yield from flat_iterator_index(dimensions, wrapped_func)


def flat_iterator_index(dimensions, func):
    partial_modulos = np.array([
        np.prod(dimensions[n+1:]) for n in range(len(dimensions))
    ])

    for flat_idx in range(np.prod(dimensions)):
        multiidx = convert_flat_index_into_multiindex(flat_idx, partial_modulos)
        yield multiidx, func(multiidx, flat_idx)


def convert_flat_index_into_multiindex(idx, partial_modulos):
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