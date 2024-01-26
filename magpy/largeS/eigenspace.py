import numpy as np
from magpy.momenta_utils import CollapseMomenta, Momenta, RestoreMomenta, Target
from ..models import Model
from .util import *
from numba import njit
from magpy.largeS.util import get_permutations, permute



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
def compute_magnon_Hamiltonian(eigvs, magnon_H_mom_space):
    order = len(eigvs)
    assert order < 13

    einsum_str = ",".join([chr(97 + i) + chr(97+13 + i) for i in range(order)])
    if order >= 1:
        einsum_str += ","
    einsum_str += "".join([chr(97 + i) for i in range(order)])
    einsum_str += "->"
    einsum_str += "".join([chr(97+13 + i) for i in range(order)])

    return np.einsum(einsum_str, *eigvs, magnon_H_mom_space)


@RestoreMomenta(
    momentum_arrays_arg_idx=0,
    output_first_momentum_idx=1,
)
@CollapseMomenta(
    momentum_arrays_arg_idx=0, 
    targets=(
        Target(arg_idx=0, first_momentum_idx=0, is_tensor=False), # k_arrays
        Target(arg_idx=1, first_momentum_idx=0, is_tensor=False), # eigvs
        Target(arg_idx=2, first_momentum_idx=1, is_tensor=True),  # magnon_Hs_mom_space
    )
)
def compute_magnon_Hamiltonians(k_arrays, eigvs, magnon_Hs_mom_space, first_momentum_idx=1):
    # if isinstance(momentum_arrays, Momenta):
    #     k_arrays = momentum_arrays.collapse()
    #     eigvs = momentum_arrays.collapse(eigvs)
    #     magnon_Hs_mom_space = momentum_arrays.collapse(magnon_Hs_mom_space, first_momentum_idx=1)
    # else: 
    #     k_arrays = momentum_arrays
    
    num_ks = np.array([len(k_array) for k_array in k_arrays], dtype=np.int64)
    last_momentum_idx = first_momentum_idx + len(k_arrays)
    magnon_Hs_shape = (
        *magnon_Hs_mom_space.shape[:first_momentum_idx], 
        *num_ks, 
        *magnon_Hs_mom_space.shape[last_momentum_idx:]
    )
    magnon_Hs = np.zeros(magnon_Hs_shape, dtype=np.complex128)

    for k_multiidx, magnon_H in iterator(
        k_arrays, lambda k_multiidx, ks: 
            compute_magnon_Hamiltonian(
                eigvs[k_multiidx], magnon_Hs_mom_space[k_multiidx])):
        idx = (*((slice(None),) * (first_momentum_idx-1)), *k_multiidx)
        magnon_Hs[idx] = magnon_H

    # if isinstance(k_arrays, Momenta):
    #     magnon_Hs = k_arrays.restore(magnon_Hs, first_momentum_idx)

    return magnon_Hs



#@njit
"""
magnon_H_eigenspace.shape == (nperm, num_bands, ..., num_bands)
where nperm = factorial(order), and num_bands appears order times
"""
def normal_order_and_symmetrize_magnon_Hamiltonian(magnon_H_eigenspace):
    order = len(magnon_H_eigenspace.shape) - 1 # 0th idx is permutation
    permutations = get_permutations(order)
    ANNIHILATOR = 0
    CREATOR = 1
    H_dim = magnon_H_eigenspace.shape[1:]
    H_normal_ordered_dim = np.array(H_dim) // 2
    
    # nosym = normal ordered and symmetrized
    magnon_H_eigenspace_nosym = np.zeros(
        (2**order, *H_normal_ordered_dim),
        dtype=np.complex128)
    commutator_terms = {}

    for ph_idx_bits in np.arange(2**order):
        # extract the individual bits representing annihilators/creators.
        # 0 = annihilator, 1 = creator
        ph_idx = np.array([
            (ph_idx_bits & 2**i) >> i \
            for i in reversed(range(order))
        ])

        for nperm_bands, perm_bands in enumerate(permutations):
            ph_idx_permuted = permute(ph_idx, perm_bands)
            index = (nperm_bands,
                    *map(lambda ph: slice(ph, None, 2), ph_idx_permuted))
            # permute band indices
            magnon_H_eigenspace_nosym_permuted = \
                np.transpose(
                    magnon_H_eigenspace[index],
                    axes=perm_bands)
            
            magnon_H_eigenspace_nosym[ph_idx_bits] += \
                magnon_H_eigenspace_nosym_permuted
            
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
                        magnon_H_eigenspace_nosym_permuted,
                        axis1=annihil_to_left_pos,
                        axis2=creator_pos)
            
        # prevent overcounting when symmetrizing
        counts = dict(zip(*np.unique(ph_idx, return_counts=True)))
        num_annihilators = counts.get(ANNIHILATOR, 0)
        num_creators = counts.get(CREATOR, 0)
        normalization_factor = np.math.factorial(num_annihilators) * \
                               np.math.factorial(num_creators)
        magnon_H_eigenspace_nosym[ph_idx_bits] /= \
            normalization_factor
        
        # TODO commutator terms
        

    return magnon_H_eigenspace_nosym



@RestoreMomenta(
    momentum_arrays_arg_idx=1,
    output_first_momentum_idx=1,
    output_is_tensor=True,
    output_restore_deep=True,
)
@CollapseMomenta(
    momentum_arrays_arg_idx=1, 
    targets=(Target(arg_idx=0, first_momentum_idx=1, is_tensor=True, collapse_deep=True),)
)
def normal_order_and_symmetrize_magnon_Hamiltonians(
        magnon_Hs_eigenspace, k_arrays=None):
    order = len(magnon_Hs_eigenspace.shape) - 2 # 0th idx is permutation, 1st idx is momenta
    num_momenta = magnon_Hs_eigenspace.shape[1]
    magnon_Hs_eigenspace_nosym = np.zeros((
        2**order,
        num_momenta,
        *(np.array(magnon_Hs_eigenspace.shape[2:]) // 2),
    ), dtype=np.complex128)
    
    for k_idx in range(num_momenta):
        magnon_Hs_eigenspace_nosym[:, k_idx] = \
            normal_order_and_symmetrize_magnon_Hamiltonian(
                magnon_Hs_eigenspace[:, k_idx])

    return magnon_Hs_eigenspace_nosym