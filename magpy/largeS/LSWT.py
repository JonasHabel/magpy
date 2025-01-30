import numpy as np
from operator import itemgetter
from magpy.models import Model
from magpy.lattice import ReciprocalLattice
from magpy.interactions import Interaction
from magpy.momenta_utils import MSQ, CollapseMomenta, Momenta, RestoreMomenta, Target
from magpy.util import BOGO_METRIC, ENERGY_EPS
from magpy.largeS import momentum_space
from magpy.largeS.util import get_real_space_magnon_Hamiltonian



def compute_LSWT_Hamiltonian_momentum_space_BdG(
        model: Model, k, LSWT_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for LSWT of one- or two-spin interactions.")
    
    if model.lattice.embedding_dim >= 1 and \
        model.lattice.embedding_dim != len(k):
        raise Exception(f"dimension of momentum vector " \
                        + f"{len(k)} must equal the "\
                        + f"embedding dimension {model.lattice.embedding_dim}")
    
    LSWT_Hamiltonian_real_space = get_real_space_magnon_Hamiltonian(
        LSWT_Hamiltonian_real_space, model, order=2)

    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    magnon_H_k, magnon_H_minusk = \
        momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
            model, ks=np.array([k]), 
            interaction_Hamiltonian_real_space=LSWT_Hamiltonian_real_space)
    
    sigma_x = np.array([[0, 1], [1, 0]])
    # of the form [a_k^†, a_{-k}] H_BdG [a_k, a_{-k}^†]
    magnon_H_BdG = np.kron(np.eye(num_sites_unit_cell), sigma_x) \
        @ (magnon_H_k + magnon_H_minusk.T)

    # make BdG Hamiltonians hermitian
    # magnon_H_BdG += np.conj(magnon_H_BdG.T)
    # magnon_H_BdG /= 2

    return magnon_H_BdG


@RestoreMomenta(
    momentum_arrays_arg_idx=1,
    output_first_momentum_idx=0,
    output_is_tensor=False,
)
@CollapseMomenta(
    targets=(Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),)
)
def compute_LSWT_Hamiltonians_momentum_space_BdG(
        model: Model, k_arrays, LSWT_Hamiltonian_real_space=None):
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    magnon_Hs_BdG = []
    
    LSWT_Hamiltonian_real_space = get_real_space_magnon_Hamiltonian(
        LSWT_Hamiltonian_real_space, model, order=2)

    for k_array in k_arrays:
        num_ks = len(k_array)
        magnon_H_BdG = np.zeros(
            (num_ks, 2*num_sites_unit_cell, 2*num_sites_unit_cell), 
            dtype=np.complex128)
        for k_idx, k in enumerate(k_array):
            magnon_H_BdG[k_idx] = \
                compute_LSWT_Hamiltonian_momentum_space_BdG(model, k=k)
        
        magnon_Hs_BdG.append(magnon_H_BdG)

    return magnon_Hs_BdG







"""
The np.linalg.eig method returns a matrix U such that U^{-1} ηH U = D is
diagonal. However, the condition U^† η U = η, which is required so that the
Bogoliubov operators obey the canonical commutation relations, does not hold
in general:
1) in the case of degenerate eigenvalues, the degenerate eigenspace is not
   necessarily orthogonal wrt the bogo metric.
2) U^† η U will be a diagonal matrix. Thus, we need to normalize the n-th
   eigenvector by sqrt((U^† η U)_{nn})

Therefore, we need to employ the Gram-Schmidt algorithm wrt the bogo metric.
"""
def orthogonalize_wrt_metric(eigv, metric):
    eigv_ortho = np.zeros(eigv.shape, dtype=complex)
    num_vec = eigv.shape[1]

    def metric_dot(a, b):
        return a.conj().T @ metric @ b

    for n in range(num_vec):
        # orthogonalize n-th vector
        collinear_component = np.sum(np.array([
            eigv_ortho[:, m] \
            * metric_dot(eigv_ortho[:, m], eigv[:, n]) \
            / metric_dot(eigv_ortho[:, m], eigv_ortho[:, m]) \
            for m in range(n)
        ]), axis=0)
        eigv_ortho[:, n] = eigv[:, n] - collinear_component
        # normalize n-th vector
        # norm_sq = metric[n, n] * metric_dot(eigv_ortho[:, n], eigv_ortho[:, n])
        # eigv_ortho[:, n] /= np.sqrt(norm_sq)

    return eigv_ortho


def normalize_wrt_metric(eigv, metric):
    eigv_normalized = np.zeros(eigv.shape, dtype=complex)
    eigv_norms = np.sqrt(np.diag(metric @ eigv.T.conj() @ metric @ eigv))
    for n in range(eigv.shape[1]):
        eigv_normalized[:, n] = eigv[:, n] / eigv_norms[n]

    return eigv_normalized


"""
Say we have eigenvalues [-2, -1, 0, 0, 1, 2].
This routine sorts the eigenvalues and corresponding eigenvectors in the order
[0, 0, 1, -1, 2, -2]. This retains the bogo-metric structure
"""
def __sort_eigensystem(eigw, eigv):
    # isolate blocks of positive, negative and zero eigenvalues and eigenvectors
    eigw_rounded = eigw.copy()
    eigw_rounded[np.abs(np.real(eigw_rounded)) < ENERGY_EPS] = 0
    pos_idx = np.where(eigw_rounded > 0)[0]
    zero_idx = np.where(eigw_rounded == 0)[0]
    neg_idx = np.where(eigw_rounded < 0)[0]

    pos_eigw, pos_eigv = eigw[pos_idx], eigv[:, pos_idx]
    zero_eigw, zero_eigv = eigw[zero_idx], eigv[:, zero_idx]
    neg_eigw, neg_eigv = eigw[neg_idx], eigv[:, neg_idx]

    # sort positive (negative) eigenvalues in ascending (descending) order
    pos_idx_sorted = pos_eigw.argsort()
    neg_idx_sorted = neg_eigw.argsort()[::-1]

    pos_eigw, pos_eigv = pos_eigw[pos_idx_sorted], pos_eigv[:, pos_idx_sorted]
    neg_eigw, neg_eigv = neg_eigw[neg_idx_sorted], neg_eigv[:, neg_idx_sorted]

    # insert positive, negative and zero blocks into sorted arrays
    eigw_sorted = np.zeros(eigw.shape, dtype=float)
    eigv_sorted = np.zeros(eigv.shape, dtype=complex)
    zero_block_idx = len(zero_eigw)

    if zero_block_idx > 0:
        print("!!! WARNING: THERE ARE GAPLESS MODES. THIS CAN LEAD TO NUMERICAL ISSUES "
              "WITH THE BDG TRAFO, E.G. PARTICLE-LIKE EIGENVECTORS WITH TINY NEGATIVE "
              "ENERGY. CONSIDER ADDING A SMALL GAP (LARGER THAN THE GLOBAL ENERGY EPSILON "
              f"{ENERGY_EPS}) !!!")
    
    if pos_eigw.shape[0] != neg_eigw.shape[0]:
        print(eigw)
        raise Exception("LSWT is not well-defined: there might be complex eigenvalues")

    eigw_sorted[0:zero_block_idx] = zero_eigw
    eigv_sorted[:, 0:zero_block_idx] = zero_eigv
    eigw_sorted[zero_block_idx::2] = pos_eigw
    eigv_sorted[:, zero_block_idx::2] = pos_eigv
    eigw_sorted[zero_block_idx+1::2] = neg_eigw
    eigv_sorted[:, zero_block_idx+1::2] = neg_eigv

    return eigw_sorted, eigv_sorted


    
"""
This is where the actual diagonalization happens
"""
def get_eigensystem_momentum_space(
        model: Model, k=None, 
        LSWT_Hamiltonian_real_space=None, 
        LSWT_Hamiltonian_momentum_space_BdG=None, 
        orthonormalize=True):
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    bogo_metric = BOGO_METRIC(num_sites_unit_cell)

    H_k = compute_LSWT_Hamiltonian_momentum_space_BdG(
            model, k, LSWT_Hamiltonian_real_space) \
        if LSWT_Hamiltonian_momentum_space_BdG is None \
        else LSWT_Hamiltonian_momentum_space_BdG

    eigw, eigv = np.linalg.eig(bogo_metric @ H_k)
    # idx = eigw.argsort()
    # eigw, eigv = eigw[idx], eigv[:, idx]
    eigw, eigv = __sort_eigensystem(eigw, eigv)

    if orthonormalize:
        eigv = orthogonalize_wrt_metric(eigv, bogo_metric)
        eigv = normalize_wrt_metric(eigv, bogo_metric)

    return eigw, eigv


@RestoreMomenta(
    momentum_arrays_arg_idx=1,
    custom_restore_func=lambda result, momenta, strip:
        tuple(MSQ(momenta.restore(x, strip=strip), momenta) for x in result)
)
@CollapseMomenta(
    targets=(Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),)
)
def get_eigensystems_momentum_space(
        model: Model, k_arrays, LSWT_Hamiltonian_real_space=None):
    # if isinstance(momenta, Momenta):
    #     assert momenta.num_momenta == 1
    #     ks = momenta.collapse()[0]
    # else: 
    #     ks = momenta
    
    LSWT_Hamiltonian_real_space = get_real_space_magnon_Hamiltonian(
        LSWT_Hamiltonian_real_space, model, order=2)

    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    eigws, eigvs = [], []

    for k_array in k_arrays:
        num_ks = len(k_array)
        eigws_for_k_array = np.zeros((num_ks, 2*num_sites_unit_cell))
        eigvs_for_k_array = np.zeros(
            (num_ks, 2*num_sites_unit_cell, 2*num_sites_unit_cell),
            dtype=np.complex128)
        for k_idx, k in enumerate(k_array):
            eigws_for_k_array[k_idx], eigvs_for_k_array[k_idx] = \
                get_eigensystem_momentum_space(
                    model, k, LSWT_Hamiltonian_real_space)
        
        eigws.append(eigws_for_k_array)
        eigvs.append(eigvs_for_k_array)
        
    # if isinstance(momenta, Momenta):
    #     eigws = momenta.restore(eigws)
    #     eigvs = momenta.restore(eigvs)

    return eigws, eigvs