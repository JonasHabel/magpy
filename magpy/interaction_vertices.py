

import numpy as np
from operator import itemgetter
from .models import Model
from .lattice import ReciprocalLattice
from .interactions import Interaction
from .util import LARGE_S_EXPANSION_COEFF
from .util_jit import prod, permute
from numba import njit


"""
returns: list of numpy array
    a list of tensors of coeff.s of the {order}-th order interaction vertices
    in real space
    e.g. order=3, returns the coefficients of
        a^† a^† a
        a^† a a
    (there are no terms like a^† a^† a^† or a a a before plugging in the
    Bogoliubov trafo)
"""
def compute_interaction_Hamiltonian_real_space(model: Model, order):
    if order not in [3] or any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for cubic vertices of one- or two-spin interactions.")
    
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    # caa = creator annihilator annihilator = a^† a a
    caa = np.zeros((2, 2, 2))
    caa[1, 0, 0] = 1
    # cca = creator creator annihilator = a^† a^† a
    cca = np.zeros((2, 2, 2))
    cca[1, 1, 0] = 1

    rotated_spin_interactions = model.compute_rotated_interactions()
    magnon_Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            site = inter.sites[0]
            magnon_BdG_tensor_iii = C[1] * (spin_int_tensor[0] * caa \
                                          + spin_int_tensor[1] * cca)
            magnon_H_iii = Interaction([site]*3, magnon_BdG_tensor_iii)
            magnon_Hamiltonians += [magnon_H_iii]
        elif len(inter.sites) == 2:
            site_i, site_j = inter.sites
            S_iii = S[site_j.subl_idx] / np.sqrt(S[site_i.subl_idx])
            S_jjj = S[site_i.subl_idx] / np.sqrt(S[site_j.subl_idx])
            S_jij = S_ijj = np.sqrt(S[site_i.subl_idx])
            S_iij = S_iji = np.sqrt(S[site_j.subl_idx])
            magnon_BdG_tensor_iii = S_iii * C[1] * (spin_int_tensor[0, 2] * caa\
                                                  + spin_int_tensor[1, 2] * cca)
            magnon_BdG_tensor_jjj = S_jjj * C[1] * (spin_int_tensor[2, 0] * caa\
                                                  + spin_int_tensor[2, 1] * cca)
            magnon_BdG_tensor_jij = -S_jij * C[0] * spin_int_tensor[0, 2] * caa
            magnon_BdG_tensor_ijj = -S_ijj * C[0] * spin_int_tensor[1, 2] * cca
            magnon_BdG_tensor_iij = -S_iij * C[0] * spin_int_tensor[2, 0] * caa
            magnon_BdG_tensor_iji = -S_iji * C[0] * spin_int_tensor[2, 1] * cca
            magnon_H_iii = Interaction([inter.sites[0]]*3, magnon_BdG_tensor_iii)
            magnon_H_jjj = Interaction([inter.sites[1]]*3, magnon_BdG_tensor_jjj)
            magnon_H_jij = Interaction([inter.sites[n] for n in [1, 0, 1]], magnon_BdG_tensor_jij)
            magnon_H_ijj = Interaction([inter.sites[n] for n in [0, 1, 1]], magnon_BdG_tensor_ijj)
            magnon_H_iij = Interaction([inter.sites[n] for n in [0, 0, 1]], magnon_BdG_tensor_iij)
            magnon_H_iji = Interaction([inter.sites[n] for n in [0, 1, 0]], magnon_BdG_tensor_iji)
            magnon_Hamiltonians += [
                magnon_H_iii, magnon_H_jjj, magnon_H_jij,
                magnon_H_ijj, magnon_H_iij, magnon_H_iji,
            ]
    
    return magnon_Hamiltonians



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
def compute_interaction_Hamiltonian_momentum_space(model: Model, order, ks,
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
        else compute_interaction_Hamiltonian_real_space(model, order)
    
    interaction_tensors_real_space, sublattice_indices, bravais_vecs = \
        __extract_real_space_Hamiltonian_quantities_for_jit(
            model, magnon_Hs_real_space)
    
    # need this case distinction for numba
    if model.lattice.dim == 0:
        return compute_interaction_Hamiltonian_momentum_space_0d_jit(order,
            interaction_tensors_real_space, sublattice_indices, bravais_vecs,
            model.lattice.num_sites_unit_cell)
    else:
        return compute_interaction_Hamiltonian_momentum_space_jit(order, ks,
            interaction_tensors_real_space, sublattice_indices, bravais_vecs,
            model.lattice.num_sites_unit_cell)


@njit
def compute_interaction_Hamiltonian_momentum_space_0d_jit(order,
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
def compute_interaction_Hamiltonian_momentum_space_jit(order, ks,
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
def compute_cubic_interaction_Hamiltonian_for_loop_momentum(model: Model,
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
        else compute_interaction_Hamiltonian_real_space(model, order=3)
    
    interaction_tensors_real_space, sublattice_indices, bravais_vecs = \
        __extract_real_space_Hamiltonian_quantities_for_jit(
            model, magnon_Hs_real_space)
    
    return compute_cubic_interaction_Hamiltonian_for_loop_momentum_jit(
        k, momenta_BZ,
        interaction_tensors_real_space, sublattice_indices, bravais_vecs,
        model.lattice.num_sites_unit_cell)
    

@njit
def GET_CUBIC_PERMUTATIONS():
    return np.array([
        [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0],
    ])

@njit
def compute_cubic_interaction_Hamiltonian_for_loop_momentum_jit(
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
                compute_interaction_Hamiltonian_momentum_space_jit(3,
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



"""
returns: list of numpy array
    tensor of coeff.s of the {order}-th order interaction vertices
    in momentum/band space (= LSWT eigenspace)
    e.g. order=3, returns the coefficients of
        α_{-ks[0]}^† α_{-ks[1]}^† α_{-k[2]}^†
        α_{-ks[0]}^† α_{-ks[1]}^† α_{k[2]}
        α_{-ks[0]}^† α_{ks[1]} α_{k[2]}
        α_{ks[0]} α_{ks[1]} α_{k[2]}
    where 
    
    Note: the eigenvectors {eigvs} must be evaluated at the same momenta as
          {interaction_Hamiltonian_mom_space} is.
    Note: momentum conservation is not enforced on this level.
    Note: the coefficients are not symmetrized.
"""
def compute_interaction_Hamiltonian_LSWT_eigenspace(model: Model, order, eigvs,
        interaction_Hamiltonian_mom_space):
    if order not in [3] or any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for cubic vertices of one- or two-spin interactions.")
    
    return compute_interaction_Hamiltonian_LSWT_eigenspace_jit(order, eigvs,
        interaction_Hamiltonian_mom_space)


@njit
def compute_interaction_Hamiltonian_LSWT_eigenspace_jit(
        order, eigvs, magnon_H_mom_space):
    magnon_H_eigenspace = np.zeros(magnon_H_mom_space.shape,
                                   dtype=np.complex128)
    H_dim = magnon_H_mom_space.shape

    if order == 3:
        for mu in range(H_dim[0]):
            for nu in range(H_dim[1]):
                for rho in range(H_dim[2]):
                    for i in range(H_dim[0]):
                        for j in range(H_dim[1]):
                            for k in range(H_dim[2]):
                                magnon_H_eigenspace[mu, nu, rho] += \
                                    eigvs[0, i, mu] \
                                  * eigvs[1, j, nu] \
                                  * eigvs[2, k, rho] \
                                  * magnon_H_mom_space[i, j, k]
                                
    return magnon_H_eigenspace




def compute_cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum(
        model: Model, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ,
        cubic_interaction_Hamiltonian_for_loop_momentum):
    num_perms = cubic_interaction_Hamiltonian_for_loop_momentum.shape[0]
    H_dim = cubic_interaction_Hamiltonian_for_loop_momentum.shape[-3:]
    Nks = eigvs_BZ.shape[:-2]
    num_bands = eigvs_BZ.shape[-2] // 2 # == eigvs_BZ.shape[-1]
    num_ks = np.prod(Nks)
    eigvs_BZ_flat = eigvs_BZ \
        .reshape((num_ks, 2*num_bands, 2*num_bands))
    eigvs_minus_k_minus_BZ_flat = eigvs_minus_k_minus_BZ \
        .reshape((num_ks, 2*num_bands, 2*num_bands))
    cubic_interaction_Hamiltonian_for_loop_momentum_flat = \
        cubic_interaction_Hamiltonian_for_loop_momentum \
        .reshape((num_perms, num_ks, *H_dim))
    
    cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum = \
    compute_cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum_jit(
            eigvs_at_k, eigvs_BZ_flat, eigvs_minus_k_minus_BZ_flat,
            cubic_interaction_Hamiltonian_for_loop_momentum_flat)
    
    return cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum \
        .reshape((num_perms, *Nks, *H_dim))


@njit
def compute_cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum_jit(
        eigvs_at_k, eigvs_BZ_flat, eigvs_minus_k_minus_BZ_flat,
        magnon_H_for_loop_momentum_flat):
    magnon_H_eigenspace_for_loop_momentum_flat = np.zeros(
        magnon_H_for_loop_momentum_flat.shape, dtype=np.complex128)
    num_perms = magnon_H_for_loop_momentum_flat.shape[0]
    num_ks = magnon_H_for_loop_momentum_flat.shape[1]
    CUBIC_PERMUTATIONS = GET_CUBIC_PERMUTATIONS()

    for nperm, permutation in enumerate(CUBIC_PERMUTATIONS):
        for nq, magnon_H_mom_space_k_q_minuskminusq, eigvs_at_q, eigvs_at_minus_k_minus_q \
            in zip(
                range(num_ks),
                magnon_H_for_loop_momentum_flat[nperm],
                eigvs_BZ_flat, eigvs_minus_k_minus_BZ_flat):
            if nq % 100 == 0:
                print("nperm = " + str(nperm) + " -- nq = " + str(nq) + " / " + str(len(eigvs_BZ_flat)))
            eigvs = np.zeros((3, *eigvs_at_k.shape), dtype=np.complex128)
            eigvs[0] = eigvs_at_minus_k_minus_q
            eigvs[1] = eigvs_at_q
            eigvs[2] = eigvs_at_k
            eigvs = permute(eigvs, permutation)
            magnon_H_eigenspace_for_loop_momentum_flat[nperm, nq] = \
                compute_interaction_Hamiltonian_LSWT_eigenspace_jit(
                    3, eigvs, magnon_H_mom_space_k_q_minuskminusq)
    
    return magnon_H_eigenspace_for_loop_momentum_flat
