import numpy as np
from magpy.momenta_utils import CollapseMomenta, RestoreMomenta, Target
from ..models import Model
from .util import *
from numba import njit
from magpy.largeS.util import get_permutations, permute
from magpy.largeS import momentum_space, eigenspace




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
                    axes=invert_permutation(perm_bands))
            
            magnon_H_eigenspace_nosym[ph_idx_bits] += \
                magnon_H_eigenspace_nosym_permuted
            
            # compute commutator terms
            creator_positions = np.array(np.where(ph_idx_permuted == CREATOR)[0])
            annihilator_positions = np.array(np.where(ph_idx_permuted == ANNIHILATOR)[0])
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
    momentum_arrays_arg_idx=0,
    output_first_momentum_idx=1,
    output_is_tensor=True,
    output_restore_deep=True,
)
@CollapseMomenta(
    targets=(Target(arg_idx=0, first_momentum_idx=1, is_tensor=True, collapse_deep=True),)
)
def normal_order_and_symmetrize_magnon_Hamiltonians(
        magnon_Hs_eigenspace):
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









def compute_commutator_term_with_permutations(
        model: Model, ks, eigvs, ks_BZ, eigvs_BZ, 
        interaction_Hamiltonian_real_space=None):
    order = len(eigvs)
    ANNIHILATOR = 0
    CREATOR = 1

    # get all permutations which do not switch the order of the second last and last indices;
    # the second last and last indices correspond to the momenta we have to trace over
    # due to the Kronecker delta from the commutators;
    # switching their order would be redundant.
    # permutations_and_inv_permutations is a list of tuples (perm, inv_perm)
    permutations_and_inv_permutations = [
        (perm, inv_perm) for perm, inv_perm in zip(
            get_permutations(num_elements = order+2),
            get_inverse_permutations(num_elements = order+2)
        ) if perm.index(order+1) > perm.index(order)
    ]
    # woco = without commuted operators
    permutations_woco = get_permutations(num_elements=order)

    H_dim = 2*model.lattice.num_sites_unit_cell
    commutator_term_shape = (np.math.factorial(order), *((H_dim,) * order))
    commutator_term = np.zeros(commutator_term_shape, dtype=np.complex128)
    
    sigma_x_ph = np.kron(np.eye(H_dim // 2), np.array([[0, 1], [1, 0]]))
    
    for k_BZ, eigv_BZ in zip(ks_BZ, eigvs_BZ):
        eigv_minus_BZ = sigma_x_ph @ eigv_BZ.conj() @ sigma_x_ph

        ks_conserved = np.array([-np.sum(ks, axis=0), *ks, k_BZ, -k_BZ]) \
            if order >= 1 else np.array([k_BZ, -k_BZ])
        eigvs_conserved = [*eigvs, eigv_BZ, eigv_minus_BZ]

        for nperm, (permutation, inv_permutation) in enumerate(permutations_and_inv_permutations):
            ks_permuted = permute(ks_conserved, permutation)
            eigvs_permuted = permute(eigvs_conserved, permutation, as_np_array=False)
            # TODO speed up using precomputed interaction_Hamiltonian_real_space
            commutator_term_loop_contrib_mom_space = \
                momentum_space.compute_magnon_Hamiltonian(
                    model, ks_permuted, interaction_Hamiltonian_real_space)
            commutator_term_loop_contrib_eigenspace = \
                eigenspace.compute_magnon_Hamiltonian(
                    eigvs_permuted, 
                    commutator_term_loop_contrib_mom_space)
            commuted_operator_idx_1 = inv_permutation[-2]
            commuted_operator_idx_2 = inv_permutation[-1]
            commuted_operators_ph_idx = (
                *((slice(None),) * commuted_operator_idx_1),
                slice(ANNIHILATOR, None, 2), # commuted_operator_idx_1
                *((slice(None),) * (commuted_operator_idx_2 - commuted_operator_idx_1 - 1)),
                slice(CREATOR, None, 2),     # commuted_operator_idx_2
            )
            
            # woco = without commuted operators
            permutation_woco = tuple(x for x in permutation if x < order)
            nperm_woco = permutations_woco.index(permutation_woco)
            commutator_term[nperm_woco] += np.trace(  # trace over band indices
                commutator_term_loop_contrib_eigenspace[commuted_operators_ph_idx],
                axis1=commuted_operator_idx_1, axis2=commuted_operator_idx_2
            )
        
    return commutator_term



@RestoreMomenta(
    momentum_arrays_arg_idx=2,
    output_first_momentum_idx=1,
    output_is_tensor=True,
)
@CollapseMomenta(
    targets=(
        Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),
        Target(arg_idx=2, first_momentum_idx=0, is_tensor=False),
        Target(arg_idx=3, first_momentum_idx=0, is_tensor=True),
        Target(arg_idx=4, first_momentum_idx=0, is_tensor=True),
    )
)
def compute_commutator_terms_with_permutations(
        model: Model, k_arrays, eigvs, ks_BZ, eigvs_BZ, 
        interaction_Hamiltonian_real_space=None):
    assert len(ks_BZ.shape) == 2 # 1st index: pos in BZ; 2nd index: momentum component (kx/ky/kz/...)
    assert len(eigvs[0]) == np.prod([len(eigv) for eigv in eigvs[1:]])

    # order is the number of boson operators in the commutator term; 
    # the actual vertex that gives rise to the commutator term contains order+2 boson operators
    order = len(k_arrays) + 1

    magnon_Hs_real_space = get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order + 2)
        
    momenta_shape = np.array([len(eigv) for eigv in eigvs], dtype=np.int64)
    H_dim = 2*model.lattice.num_sites_unit_cell
    commutator_terms_shape = \
        (np.math.factorial(order), *momenta_shape, *((H_dim,) * order))
    commutator_terms = \
        np.zeros(commutator_terms_shape, dtype=np.complex128)
    
    def compute_commutator_term(k_multiidx, k_flat_idx, ks):
        eigv_multiidx = np.array([k_flat_idx, *k_multiidx], dtype=np.int64)
        return compute_commutator_term_with_permutations(
            model,
            ks, get_quantities_at_multiidx(eigvs, eigv_multiidx),
            ks_BZ, eigvs_BZ,
            magnon_Hs_real_space)
    
    for k_multiidx, commutator_term in flat_iterator(
            k_arrays, (model.lattice.dim,), compute_commutator_term):
        commutator_terms[(slice(None), *k_multiidx)] = commutator_term

    return commutator_terms
    
