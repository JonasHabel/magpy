import numpy as np
from magpy.momenta_utils import CollapseMomenta, RestoreMomenta, Target
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
    assert order <= 26

    einsum_str = ",".join([chr(97 + i) + chr(65 + i) for i in range(order)])
    if order >= 1:
        einsum_str += ","
    einsum_str += "".join([chr(97 + i) for i in range(order)])
    einsum_str += "->"
    einsum_str += "".join([chr(65 + i) for i in range(order)])

    return np.einsum(einsum_str, *eigvs, magnon_H_mom_space)



def compute_magnon_Hamiltonian_with_permutations(eigvs, magnon_H_mom_space):
    order = len(eigvs)
    magnon_H = np.zeros(magnon_H_mom_space.shape, dtype=np.complex128)

    for nperm, (perm, magnon_H_mom_space_for_perm) in enumerate(zip(
        get_permutations(order), magnon_H_mom_space
    )):
        eigvs_permuted = permute(eigvs, perm)
        magnon_H[nperm] = compute_magnon_Hamiltonian(
            eigvs_permuted, magnon_H_mom_space_for_perm)
        
    return magnon_H


@RestoreMomenta(
    momentum_arrays_arg_idx=2,
    output_first_momentum_idx=1,
)
@CollapseMomenta(
    targets=(
        Target(arg_idx=1, first_momentum_idx=0, is_tensor=False), # eigvs
        Target(arg_idx=2, first_momentum_idx=1, is_tensor=True),  # magnon_Hs_mom_space
    )
)
def compute_magnon_Hamiltonians_with_permutations(model: Model, eigvs, magnon_Hs_mom_space):
    # if isinstance(momentum_arrays, Momenta):
    #     k_arrays = momentum_arrays.collapse()
    #     eigvs = momentum_arrays.collapse(eigvs)
    #     magnon_Hs_mom_space = momentum_arrays.collapse(magnon_Hs_mom_space, first_momentum_idx=1)
    # else: 
    #     k_arrays = momentum_arrays
    
    order = len(eigvs)
    first_momentum_idx = 1
    last_momentum_idx = first_momentum_idx + order - 1
    num_ks = magnon_Hs_mom_space.shape[first_momentum_idx:last_momentum_idx]
    assert len(num_ks) == order - 1
    assert eigvs[0].shape[0] == np.prod([eigvs[n+1].shape[0] for n in range(0, len(num_ks))])
    magnon_Hs_shape = (
        *magnon_Hs_mom_space.shape[:first_momentum_idx], # permutations
        *num_ks,                                         # momenta
        *magnon_Hs_mom_space.shape[last_momentum_idx:]   # bands 
    )
    magnon_Hs = np.zeros(magnon_Hs_shape, dtype=np.complex128)

    def compute_magnon_H(k_multiidx, k_flat_idx):
        idx = (slice(None), *k_multiidx)
        return compute_magnon_Hamiltonian_with_permutations(
            np.array(
                [eigvs[0][k_flat_idx]] + \
                [eigvs[n+1][k_multiidx[n]] for n in range(0, len(num_ks))]
            ),
            magnon_Hs_mom_space[idx])

    for k_multiidx, magnon_H in flat_iterator_index(
            num_ks, compute_magnon_H):
        idx = (slice(None), *k_multiidx)
        magnon_Hs[idx] = magnon_H

    # if isinstance(k_arrays, Momenta):
    #     magnon_Hs = k_arrays.restore(magnon_Hs, first_momentum_idx)

    return magnon_Hs



