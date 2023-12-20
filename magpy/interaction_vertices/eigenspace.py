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
    order = 3
    PERMUTATIONS = GET_PERMUTATIONS(order)
    ANNIHILATOR = 0
    CREATOR = 1
    loop_mom_shape = magnon_H_eigenspace_for_loop_momentum_flat.shape[1:-order]
    num_loop_momenta = len(loop_mom_shape)
    H_dim = magnon_H_eigenspace_for_loop_momentum_flat.shape[-order:]
    H_normal_ordered_dim = np.array(H_dim) // 2
    
    # nosym = normal ordered and symmetrized
    magnon_H_eigenspace_nosym_for_loop_momentum_flat = np.zeros(
        (2**order, *loop_mom_shape, *H_normal_ordered_dim),
        dtype=np.complex128)
    commutator_terms = {}
    np.zeros(
        (order, *loop_mom_shape[1:], H_normal_ordered_dim[0]),
        dtype=np.complex128)

    for ph_idx_bits in np.arange(2**order):
        # extract the individual bits representing annihilators/creators.
        # 0 = annihilator, 1 = creator
        ph_idx = np.array([
            (ph_idx_bits & 2**i) >> i \
            for i in reversed(range(order))
        ])

        for nperm_bands, perm_bands in enumerate(PERMUTATIONS):
            ph_idx_permuted = permute(ph_idx, perm_bands)
            index = (nperm_bands,
                    *((slice(None),)*num_loop_momenta),
                    *map(lambda ph: slice(ph, None, 2), ph_idx_permuted))
            # keep BZ momentum index and permute band indices
            momentum_permutation = np.arange(num_loop_momenta)
            band_permutation = [x+num_loop_momenta for x in perm_bands]
            magnon_H_eigenspace_nosym_for_loop_momentum_permuted = \
                np.transpose(
                    magnon_H_eigenspace_for_loop_momentum_flat[index],
                    axes=(*momentum_permutation, *band_permutation))
            
            magnon_H_eigenspace_nosym_for_loop_momentum_flat[ph_idx_bits] += \
                magnon_H_eigenspace_nosym_for_loop_momentum_permuted
            
            # compute commutator terms
            creator_positions = np.array(np.where(ph_idx == CREATOR)[0])
            annihilator_positions = np.array(np.where(ph_idx == ANNIHILATOR)[0])
            for ncreator, creator_pos in enumerate(creator_positions):
                # for all annihilators left of the creator at {creator_pos},
                # commute them with the creator. 
                annihilator_to_left_positions = annihilator_positions[
                    np.where(annihilator_positions < creator_pos)]
                for annihil_to_left_pos in annihilator_to_left_positions:
                    # trace over bands
                    commutator_term = np.trace(
                        magnon_H_eigenspace_nosym_for_loop_momentum_permuted,
                        axis1=num_loop_momenta+annihil_to_left_pos,
                        axis2=num_loop_momenta+creator_pos)
            
        # prevent overcounting when symmetrizing
        normalization_factor = factorial(count(ph_idx, ANNIHILATOR)) * \
                               factorial(count(ph_idx, CREATOR))
        magnon_H_eigenspace_nosym_for_loop_momentum_flat[ph_idx_bits] /= \
            normalization_factor
        

    return magnon_H_eigenspace_nosym_for_loop_momentum_flat