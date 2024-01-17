import numpy as np
import itertools
from magpy.models import Model
from magpy.largeS import real_space
from magpy.largeS.util import get_real_space_magnon_Hamiltonian




def compute_magnon_Hamiltonian(model: Model, ks,
        interaction_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and model.lattice.dim != ks.shape[-1]:
        raise Exception(f"dimension of each momentum vector " \
                      + f"{ks.shape[-1]} must equal the " \
                      + f"lattice dimension {model.lattice.dim}")
    
    order = ks.shape[0]

    magnon_Hs_real_space = get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order)

    H_dim = 2*model.lattice.num_sites_unit_cell
    magnon_H_mom_space = np.zeros((H_dim,) * order, dtype=np.complex128)

    ks = ks.astype(np.float64)
    for coupling in magnon_Hs_real_space:
        subl_idxs = [site.subl_idx for site in coupling.sites]
        bravais_vecs = np.array([
            model.lattice.to_canonical_basis(site.bravais_coords) \
            for site in coupling.sites
        ]).reshape(ks.shape)    # reshape only necessary for order == 0 case
        phase = np.exp(1j * np.einsum("ij,ij", ks, bravais_vecs))
        idx = tuple(slice(2*subl_idx, 2*(subl_idx+1)) for subl_idx in subl_idxs)
        magnon_H_mom_space[idx] += phase * coupling.interaction_tensor
        
    # make BdG Hamiltonians hermitian
    # magnon_H_mom_space += \
    #     np.conj(np.transpose(magnon_H_mom_space, axes=[0, 2, 1]))
    # magnon_H_mom_space /= 2

    return magnon_H_mom_space



def compute_magnon_Hamiltonian_with_momentum_conservation(model: Model, ks,
        interaction_Hamiltonian_real_space=None):
    return compute_magnon_Hamiltonian(
        model,
        np.array([-np.sum(ks, axis=0), *ks]),
        interaction_Hamiltonian_real_space)


def compute_magnon_Hamiltonians_with_momentum_conservation(
        model: Model, k_arrays,
        interaction_Hamiltonian_real_space=None):
    if len(k_arrays) >= 1 and any(map(lambda k_array: k_array.shape[-1] != model.lattice.dim, k_arrays)):
        raise Exception(f"dimensions of momentum vectors " \
                        + f"{[k_array[-1].shape for k_array in k_arrays]} " \
                        + f"must equal the lattice dimension " \
                        + f"{model.lattice.dim}")
    
    order = len(k_arrays) + 1
    magnon_Hs_real_space = get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order)
    
    num_ks = np.array([len(k_array) for k_array in k_arrays], dtype=np.int64)
    partial_modulos = np.array([
        np.prod(num_ks[n+1:]) for n in range(len(num_ks))
    ])
    permutations = get_permutations(num_elements = order)
    num_permutations = len(permutations)
    H_dim = 2*model.lattice.num_sites_unit_cell
    magnon_Hs_shape = (num_permutations, *num_ks, *((H_dim,) * order))
    magnon_Hs = np.zeros(magnon_Hs_shape, dtype=np.complex128)

    for nperm, permutation in enumerate(permutations):
        for i in range(np.prod(num_ks)):
            k_multiidx = convert_1d_index_into_multiindex(i, partial_modulos)
            unique_ks = np.array([
                k_array[k_multiidx[n]] for n, k_array in enumerate(k_arrays)
            ]).reshape((order - 1, model.lattice.dim))   # reshaping required only if order == 1
            ks = np.array([-np.sum(unique_ks, axis=0), *unique_ks])
            ks = permute(ks, permutation)
            magnon_Hs[(nperm, *k_multiidx)] = \
                compute_magnon_Hamiltonian(model, ks, magnon_Hs_real_space)
    
    return magnon_Hs
    


def get_permutations(num_elements):
    return list(itertools.permutations(range(num_elements)))


def permute(arr, perm):
    return np.array([arr[perm[i]] for i in range(len(arr))])


def convert_1d_index_into_multiindex(idx, partial_modulos):
    quotient, remainder = (0, idx)
    multiidx = np.zeros(len(partial_modulos), dtype=np.int64)
    for n, N in enumerate(partial_modulos):
        quotient, remainder = remainder // N, remainder % N
        multiidx[n] = quotient
        
    return multiidx