import numpy as np
from ..models import Model
from ..util_jit import permute, permute_all, factorial, count
from .util import *
from numba import njit



"""
returns: list of numpy array
    tensor of coeff.s of the {order}-th order interaction vertices
    in momentum/band space (= LSWT eigenspace)
    e.g. order=3, returns the coefficients of
        α_{ks[0]} α_{ks[1]} α_{k[2]}
        α_{-ks[0]} α_{ks[1]} α_{k[2]}^†
        α_{-ks[0]} α_{ks[1]}^† α_{k[2]}
        α_{-ks[0]}^† α_{ks[1]} α_{k[2]}
        α_{-ks[0]} α_{-ks[1]}^† α_{k[2]}^†
        α_{-ks[0]}^† α_{-ks[1]} α_{k[2]}^†
        α_{-ks[0]}^† α_{-ks[1]}^† α_{k[2]}
        α_{-ks[0]}^† α_{-ks[1]}^† α_{-k[2]}^†
    where 
    
    Note: the eigenvectors {eigvs} must be evaluated at the same momenta as
          {interaction_Hamiltonian_mom_space} is.
    Note: momentum conservation is not enforced on this level.
    Note: the coefficients are not symmetrized.
"""
def compute_interaction_Hamiltonian(model: Model, order, eigvs,
        interaction_Hamiltonian_mom_space):
    if order not in [3] or any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for cubic vertices of one- or two-spin interactions.")
    
    return compute_interaction_Hamiltonian_jit(order, eigvs,
        interaction_Hamiltonian_mom_space)


@njit
def compute_interaction_Hamiltonian_jit(
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




def compute_cubic_interaction_Hamiltonian_loop(
        model: Model, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ,
        cubic_interaction_Hamiltonian_for_loop_momentum,
        normal_order_and_symmetrize=False):
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
        compute_cubic_interaction_Hamiltonian_loop_jit(
            eigvs_at_k, eigvs_BZ_flat, eigvs_minus_k_minus_BZ_flat,
            cubic_interaction_Hamiltonian_for_loop_momentum_flat)
    
    if normal_order_and_symmetrize:
        cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum = \
            normal_order_and_symmetrize_cubic_interaction_Hamiltonian_loop_jit(
                cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum)
    
    return cubic_interaction_Hamiltonian_LSWT_eigenspace_for_loop_momentum \
        .reshape((num_perms, *Nks, *H_dim))


@njit
def compute_cubic_interaction_Hamiltonian_loop_jit(
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
                compute_interaction_Hamiltonian_jit(
                    3, eigvs, magnon_H_mom_space_k_q_minuskminusq)
    
    return magnon_H_eigenspace_for_loop_momentum_flat



#@njit
def normal_order_and_symmetrize_cubic_interaction_Hamiltonian_loop_jit(
        magnon_H_eigenspace_for_loop_momentum_flat):
    CUBIC_PERMUTATIONS = GET_CUBIC_PERMUTATIONS()
    ANNIHILATOR = 0
    CREATOR = 1
    all_ph_idxs = [
        np.array([
            [ANNIHILATOR, ANNIHILATOR, ANNIHILATOR],
        ]),
        np.array([
            [CREATOR, ANNIHILATOR, ANNIHILATOR],
            [ANNIHILATOR, CREATOR, ANNIHILATOR],
            [ANNIHILATOR, ANNIHILATOR, CREATOR],
        ]),
        np.array([
            [CREATOR, CREATOR, ANNIHILATOR],
            [CREATOR, ANNIHILATOR, CREATOR],
            [ANNIHILATOR, CREATOR, CREATOR],
        ]),
        np.array([
            [CREATOR, CREATOR, CREATOR],
        ]),
    ]
    num_perms = magnon_H_eigenspace_for_loop_momentum_flat.shape[0]
    num_qs = magnon_H_eigenspace_for_loop_momentum_flat.shape[1]
    H_dim = magnon_H_eigenspace_for_loop_momentum_flat.shape[-3:]
    H_normal_ordered_dim = np.array(H_dim) // 2
    # nosym = normal ordered and symmetrized
    magnon_H_eigenspace_nosym_for_loop_momentum_flat = [
        [
            np.zeros((num_qs, *H_normal_ordered_dim), dtype=np.complex128) \
            for _ in range(len(ph_idxs))
        ] for ph_idxs in all_ph_idxs
    ]

    for nph_idxs, ph_idxs in enumerate(all_ph_idxs):
        for nph_idx, ph_idx in enumerate(ph_idxs):
            for nperm_bands, perm_bands in enumerate(CUBIC_PERMUTATIONS):
                ph_idx_permuted = permute(ph_idx, perm_bands)
                index = (nperm_bands,
                        slice(None),
                        *map(lambda ph: slice(ph, None, 2), ph_idx_permuted))
                # keep BZ momentum index and permute band indices
                band_permutation = [x+1 for x in perm_bands]
                magnon_H_eigenspace_nosym_for_loop_momentum_permuted = \
                    np.transpose(
                        magnon_H_eigenspace_for_loop_momentum_flat[index],
                        axes=(0, *band_permutation))
                
                magnon_H_eigenspace_nosym_for_loop_momentum_flat[nph_idxs][nph_idx] += \
                    magnon_H_eigenspace_nosym_for_loop_momentum_permuted
            
        # prevent overcounting when symmetrizing
        normalization_factor = factorial(count(ph_idx, ANNIHILATOR)) * \
                               factorial(count(ph_idx, CREATOR))
        for nph_idx, ph_idx in enumerate(ph_idxs):
            magnon_H_eigenspace_nosym_for_loop_momentum_flat[nph_idxs][nph_idx] /= \
                normalization_factor

    return magnon_H_eigenspace_nosym_for_loop_momentum_flat