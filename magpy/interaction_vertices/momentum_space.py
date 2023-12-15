import numpy as np
from ..models import Model
from . import real_space
from ..util_jit import prod, permute
from .util import GET_CUBIC_PERMUTATIONS
from numba import njit


"""
returns: list of numpy array
    tensor of coeff.s of the {order}-th order interaction vertices
    in momentum/sublattice space
    e.g. order=3, returns the coefficients of
        a_{-ks[0]}^† a_{-ks[1]}^† a_{k[2]}
        a_{-ks[0]}^† a_{ks[1]} a_{k[2]}
    (there are no terms like a^† a^† a^† or a a a before plugging in the
    Bogoliubov trafo)
    
    Note: momentum conservation is not enforced on this level.
    Note: the coefficients are not symmetrized.
"""
def compute_interaction_Hamiltonian(model: Model, order, ks,
        interaction_Hamiltonian_real_space=None):
    if order not in [3] or any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for cubic vertices of one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and \
        model.lattice.dim != ks.shape[-1]:
        dim = ks.shape[-1]
        raise Exception(f"dimension of momentum vectors " \
                        + f"{dim} must equal the "\
                        + f"lattice dimension {model.lattice.dim}")

    magnon_Hs_real_space = interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_interaction_Hamiltonian(model, order)
    
    interaction_tensors_real_space, sublattice_indices, bravais_vecs = \
        __extract_real_space_Hamiltonian_quantities_for_jit(
            model, magnon_Hs_real_space)
    
    # need this case distinction for numba
    if model.lattice.dim == 0:
        return compute_interaction_Hamiltonian_0d_jit(order,
            interaction_tensors_real_space, sublattice_indices, bravais_vecs,
            model.lattice.num_sites_unit_cell)
    else:
        return compute_interaction_Hamiltonian_jit(order, ks,
            interaction_tensors_real_space, sublattice_indices, bravais_vecs,
            model.lattice.num_sites_unit_cell)


@njit
def compute_interaction_Hamiltonian_0d_jit(order,
        interaction_tensors_real_space, sublattice_indices, bravais_vecs,
        num_sites_unit_cell):
    H_dim = 2*num_sites_unit_cell
    if order == 3:
        # numba doesn't support dynamic tuple creation -> need this if statement
        magnon_H_mom_space = np.zeros((H_dim, H_dim, H_dim),
                                      dtype=np.complex128)

    bravais_vecs = bravais_vecs.astype(np.float64)
    for interaction_tensor, subl_idxs_for_inter, bravais_vecs_for_inter in zip(
            interaction_tensors_real_space, sublattice_indices, bravais_vecs):
        if order == 3:
            magnon_H_mom_space[
                2*subl_idxs_for_inter[0]:2*(subl_idxs_for_inter[0]+1),
                2*subl_idxs_for_inter[1]:2*(subl_idxs_for_inter[1]+1),
                2*subl_idxs_for_inter[2]:2*(subl_idxs_for_inter[2]+1),
            ] += interaction_tensor

    return magnon_H_mom_space


@njit
def compute_interaction_Hamiltonian_jit(order, ks,
        interaction_tensors_real_space, sublattice_indices, bravais_vecs,
        num_sites_unit_cell):
    H_dim = 2*num_sites_unit_cell
    if order == 3:
        # numba doesn't support dynamic tuple creation -> need this if statement
        magnon_H_mom_space = np.zeros((H_dim, H_dim, H_dim),
                                      dtype=np.complex128)

    # numba can't deal with dot(array(int), array(int))
    # so need to convert to float64 first
    ks = ks.astype(np.float64)
    bravais_vecs = bravais_vecs.astype(np.float64)
    for interaction_tensor, subl_idxs_for_inter, bravais_vecs_for_inter in zip(
            interaction_tensors_real_space, sublattice_indices, bravais_vecs):
        if order == 3:
            phase = np.exp(1j*np.dot(ks[0], bravais_vecs_for_inter[0])
                         + 1j*np.dot(ks[1], bravais_vecs_for_inter[1])
                         + 1j*np.dot(ks[2], bravais_vecs_for_inter[2]))
            magnon_H_mom_space[
                2*subl_idxs_for_inter[0]:2*(subl_idxs_for_inter[0]+1),
                2*subl_idxs_for_inter[1]:2*(subl_idxs_for_inter[1]+1),
                2*subl_idxs_for_inter[2]:2*(subl_idxs_for_inter[2]+1),
            ] += phase * interaction_tensor
        
    # make BdG Hamiltonians hermitian
    # magnon_H_mom_space += \
    #     np.conj(np.transpose(magnon_H_mom_space, axes=[0, 2, 1]))
    # magnon_H_mom_space /= 2

    return magnon_H_mom_space






"""
returns: numpy array
    shape is (6, *Nks_BZ, num_bands, num_bands, num_bands)
    tensors of coeff.s of cubic interaction vertices for every permutation of
    [-k-q, q, k], where q runs over the whole Brioullin zone
"""
def compute_cubic_interaction_Hamiltonian_loop(model: Model,
        k, momenta_BZ, interaction_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and \
       (model.lattice.dim != k.shape[-1] or \
        model.lattice.dim != momenta_BZ.shape[-1]):
        dim1, dim2 = k.shape[-1], momenta_BZ.shape[-1]
        raise Exception(f"dimension of momentum vectors " \
                        + f"{dim1}/{dim2} must equal the "\
                        + f"lattice dimension {model.lattice.dim}")

    magnon_Hs_real_space = interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_interaction_Hamiltonian(model, order=3)
    
    interaction_tensors_real_space, sublattice_indices, bravais_vecs = \
        __extract_real_space_Hamiltonian_quantities_for_jit(
            model, magnon_Hs_real_space)
    
    return compute_cubic_interaction_Hamiltonian_loop_jit(
        k, momenta_BZ,
        interaction_tensors_real_space, sublattice_indices, bravais_vecs,
        model.lattice.num_sites_unit_cell)
    

@njit
def compute_cubic_interaction_Hamiltonian_loop_jit(
        k, momenta_BZ,
        interaction_tensors_real_space, sublattice_indices, bravais_vecs,
        num_sites_unit_cell):
    dim = momenta_BZ.shape[-1]
    Nks = momenta_BZ.shape[:-1]
    num_ks = prod(Nks)
    num_bands = 2*num_sites_unit_cell
    momenta_BZ_flat = np.ascontiguousarray(momenta_BZ).reshape((num_ks, dim))
    CUBIC_PERMUTATIONS = GET_CUBIC_PERMUTATIONS()

    magnon_H_loop_mom = np.zeros(
        (len(CUBIC_PERMUTATIONS), num_ks, num_bands, num_bands, num_bands),
        dtype=np.complex128)

    for nperm, permutation in enumerate(CUBIC_PERMUTATIONS):
        for nq, q in enumerate(momenta_BZ_flat):
            ks = np.zeros((3, len(k)))
            ks[0], ks[1], ks[2] = -k-q, q, k
            ks = permute(ks, permutation)
            magnon_H_loop_mom[nperm, nq] = \
                compute_interaction_Hamiltonian_jit(3,
                    ks, interaction_tensors_real_space, sublattice_indices,
                    bravais_vecs, num_sites_unit_cell)
            # magnon_H_k_q_kminusq_perm = \
            #     compute_interaction_Hamiltonian_momentum_space_jit(3,
            #         ks, interaction_tensors_real_space, sublattice_indices,
            #         bravais_vecs, num_sites_unit_cell)
            # magnon_H_k_BZ_kminusBZ[nperm, nq] = np.transpose(
            #     magnon_H_k_q_kminusq_perm, axes=permutation
            # )
    
    return magnon_H_loop_mom.reshape(
        (len(CUBIC_PERMUTATIONS), *Nks, num_bands, num_bands, num_bands))



def __extract_real_space_Hamiltonian_quantities_for_jit(
        model: Model, magnon_Hs_real_space):
    interaction_tensors_real_space = np.array(list(map(
        lambda coupling: coupling.interaction_tensor,
        magnon_Hs_real_space)))
    sublattice_indices = np.array(list(map(
        lambda coupling: [site.subl_idx for site in coupling.sites],
        magnon_Hs_real_space)))
    bravais_vecs = np.array(list(map(
        lambda coupling: [
            model.lattice.to_canonical_basis(site.bravais_coords) \
            for site in coupling.sites
        ],
        magnon_Hs_real_space)))
    
    return interaction_tensors_real_space, sublattice_indices, bravais_vecs
