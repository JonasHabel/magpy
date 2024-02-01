import numpy as np
from magpy.largeS.util import get_inverse_permutations, get_permutations, permute
from magpy.models import Model
from magpy.largeS import real_space
from magpy.largeS.util import get_real_space_magnon_Hamiltonian, flat_iterator
from magpy.momenta_utils import CollapseMomenta, Momenta, RestoreMomenta, Target




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

    return magnon_H_mom_space



def compute_magnon_Hamiltonian_with_momentum_conservation(model: Model, ks,
        interaction_Hamiltonian_real_space=None):
    return compute_magnon_Hamiltonian(
        model,
        np.array([-np.sum(ks, axis=0), *ks]),
        interaction_Hamiltonian_real_space)


def compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
        model: Model, ks, interaction_Hamiltonian_real_space=None):
    order = len(ks) + 1
    permutations = get_permutations(num_elements = order)
    num_permutations = len(permutations)
    H_dim = 2*model.lattice.num_sites_unit_cell
    magnon_Hs_shape = (num_permutations, *((H_dim,) * order))
    magnon_Hs = np.zeros(magnon_Hs_shape, dtype=np.complex128)
    ks_conserved = np.array([-np.sum(ks, axis=0), *ks])
    
    for nperm, permutation in enumerate(permutations):
        ks_permuted = permute(ks_conserved, permutation)
        magnon_Hs[nperm] = compute_magnon_Hamiltonian(
            model, ks_permuted, interaction_Hamiltonian_real_space)
        
    return magnon_Hs


@RestoreMomenta(
    momentum_arrays_arg_idx=1,
    output_first_momentum_idx=1,
    output_is_tensor=True,
)
@CollapseMomenta(
    targets=(Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),)
)
def compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
        model: Model, k_arrays,
        interaction_Hamiltonian_real_space=None):
#    if isinstance(momentum_arrays, Momenta):
#        k_arrays = momentum_arrays.collapse()
#    else:
#        k_arrays = momentum_arrays

    if len(k_arrays) >= 1 and any(map(lambda k_array: k_array.shape[-1] != model.lattice.dim, k_arrays)):
        raise Exception(f"dimensions of momentum vectors " \
                        + f"{[k_array[-1].shape for k_array in k_arrays]} " \
                        + f"must equal the lattice dimension " \
                        + f"{model.lattice.dim}")
    
    order = len(k_arrays) + 1
    magnon_Hs_real_space = get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order)
    
    num_ks = np.array([len(k_array) for k_array in k_arrays], dtype=np.int64)
    H_dim = 2*model.lattice.num_sites_unit_cell
    magnon_Hs_shape = (np.math.factorial(order), *num_ks, *((H_dim,) * order))
    magnon_Hs = np.zeros(magnon_Hs_shape, dtype=np.complex128)

    def compute_magnon_H(k_multiidx, k_flat_idx, ks):
        return compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
            model, ks, magnon_Hs_real_space)

    for k_multiidx, magnon_H in flat_iterator(
            k_arrays, (model.lattice.dim,), compute_magnon_H):
        magnon_Hs[(slice(None), *k_multiidx)] = magnon_H

#    if isinstance(momentum_arrays, Momenta):
#        magnon_Hs = momentum_arrays.restore(magnon_Hs, first_momentum_idx=1)

    return magnon_Hs
    





def compute_commutator_term_with_momentum_conservation_and_permutations(
        model: Model, ks, ks_BZ, interaction_Hamiltonian_real_space=None):
    order = len(ks) + 1
    ANNIHILATOR = 0
    CREATOR = 1

    # get all permutations which do not switch the order of the second last and last indices;
    # the second last and last indices correspond to the momenta we have to trace over
    # due to the Kronecker delta from the commutators;
    # switching their order would be redundant.
    # permutations_and_inv_permutations is a list of tuples (perm, inv_perm)
    permutations_and_inv_permutations = filter(
        lambda perm, inv_perm: perm.index(order) > perm.index(order-1),
        zip(
            get_permutations(num_elements = order+1),
            get_inverse_permutations(num_elements = order+1)
        )
    )

    num_permutations = len(permutations_and_inv_permutations)
    H_dim = 2*model.lattice.num_sites_unit_cell
    commutator_term_shape = (num_permutations, *((H_dim,) * order))
    commutator_term = np.zeros(commutator_term_shape, dtype=np.complex128)
    
    for k_BZ in ks_BZ:
        ks_conserved = np.array([-np.sum(ks, axis=0), *ks, k_BZ, -k_BZ])
        for nperm, (permutation, inv_permutation) in enumerate(zip(*permutations_and_inv_permutations)):
            ks_permuted = permute(ks_conserved, permutation)
            commutator_term_loop_contribution = \
                compute_magnon_Hamiltonian(
                    model, ks_permuted, interaction_Hamiltonian_real_space)
            commuted_operator_idx_1 = inv_permutation[-2]
            commuted_operator_idx_2 = inv_permutation[-1]
            commuted_operators_ph_idx = (
                *((slice(None),) * commuted_operator_idx_1),
                slice(None, None, ANNIHILATOR), # commuted_operator_idx_1
                *((slice(None),) * (commuted_operator_idx_2 - commuted_operator_idx_1 - 1)),
                slice(None, None, CREATOR),     # commuted_operator_idx_2
            )
            
            commutator_term[nperm] += np.trace(     # trace over bands
                commutator_term_loop_contribution[commuted_operators_ph_idx],
                axis1=commuted_operator_idx_1, axis2=commuted_operator_idx_2
            )
        
    return commutator_term



@RestoreMomenta(
    momentum_arrays_arg_idx=1,
    output_first_momentum_idx=1,
    output_is_tensor=True,
)
@CollapseMomenta(
    targets=(
        Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),
        Target(arg_idx=2, first_momentum_idx=0, is_tensor=True),
    )
)
def compute_commutator_terms_with_momentum_conservation_and_permutations(
        model: Model, k_arrays, ks_BZ, interaction_Hamiltonian_real_space=None):
    assert len(ks_BZ.shape) == 2 # 1st index: pos in BZ; 2nd index: momentum component (kx/ky/kz/...)

    # order is the number of boson operators in the commutator term; 
    # the actual vertex that gives rise to the commutator term contains order+2 boson operators
    order = len(k_arrays) + 1

    magnon_Hs_real_space = get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order)
        
    num_ks = np.array([len(k_array) for k_array in k_arrays], dtype=np.int64)
    H_dim = 2*model.lattice.num_sites_unit_cell
    commutator_terms_shape = \
        (np.math.factorial(order), *num_ks, *((H_dim,) * order))
    commutator_terms = \
        np.zeros(commutator_terms_shape, dtype=np.complex128)
    
    def compute_commutator_term(k_multiidx, k_flat_idx, ks):
        return compute_commutator_term_with_momentum_conservation_and_permutations(
            model, ks, ks_BZ, magnon_Hs_real_space)
    
    for k_multiidx, commutator_term in flat_iterator(
            k_arrays, (model.lattice.dim,), compute_commutator_term):
        commutator_terms[(slice(None), *k_multiidx)] = commutator_term

    return commutator_terms
    
