
# Linear spin wave theory

import numpy as np
from operator import itemgetter
from .models import Model
from .lattice import ReciprocalLattice
from .interactions import Interaction
from .util import BOGO_METRIC, LARGE_S_EXPANSION_COEFF




"""
model: Model
order: integer
    order of the vertex factors in 1/sqrt(S)
    e.g. order=2 -> LSWT, O(S); order=3 -> O(sqrt(S)); ...

We use the Holstein-Primakoff mapping. Each spin S' is mapped as
    S_i^z = S - a_i^dagger a_i
    S_i^+ = sqrt(2S - a_i^dagger a_i) a_i
    S_i^- = a_i^dagger sqrt(2S - a_i^dagger)
The square root terms are then expanded in powers of 1/sqrt(S):
    sqrt(2S - a_i^dagger a_i) = sqrt(2S) - a_i^dagger a_i/sqrt(8S) + ...
and terms of order <code>order</code> are then collected.

returns: list of Interaction
    each interaction is specified by a BdG-type interaction_tensor
    e.g. order=2 -> [a_i, a_i^dagger] H_bdg [a_j a_j^dagger]
"""
def compute_LSWT_Hamiltonian_real_space(model: Model):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    # ca = creator annihilator = a_i^† a_i
    ca = np.array([[0, 0], [1, 0]])

    rotated_spin_interactions = model.compute_rotated_interactions()
    LSWT_Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            site = inter.sites[0]
            magnon_BdG_tensor_ii = -spin_int_tensor[2] * ca
            magnon_H_ii = Interaction([site]*2, magnon_BdG_tensor_ii)
            LSWT_Hamiltonians += [magnon_H_ii]
        elif len(inter.sites) == 2:
            site_i, site_j = inter.sites
            S_ij = np.sqrt(S[site_i.subl_idx] * S[site_j.subl_idx])
            S_ii = S[site_j.subl_idx]
            S_jj = S[site_i.subl_idx]
            magnon_BdG_tensor_ij =  S_ij * C[0]**2 * spin_int_tensor[0:2, 0:2]
            magnon_BdG_tensor_ii = -S_ii * spin_int_tensor[2, 2] * ca
            magnon_BdG_tensor_jj = -S_jj * spin_int_tensor[2, 2] * ca
            magnon_H_ij = Interaction(inter.sites, magnon_BdG_tensor_ij)
            magnon_H_ii = Interaction([site_i]*2, magnon_BdG_tensor_ii)
            magnon_H_jj = Interaction([site_j]*2, magnon_BdG_tensor_jj)
            LSWT_Hamiltonians += [
                magnon_H_ij, magnon_H_ii, magnon_H_jj,
            ]   
    
    return LSWT_Hamiltonians


"""
returns: numpy array
    the momentum space BdG Hamiltonians along the k-path
"""
def compute_LSWT_Hamiltonian_along_momentum_path(model: Model,
        momentum_path: ReciprocalLattice.MomentumPath):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for LSWT of one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and \
        model.lattice.dim != momentum_path.momenta.shape[1]:
        dim = momentum_path.momenta.shape[1]
        raise Exception(f"dimension of momentum vectors " \
                        + f"{dim} must equal the "\
                        + f"lattice dimension {model.lattice.dim}")

    order = 2   # LSWT
    magnon_Hs_real_space = compute_LSWT_Hamiltonian_real_space(model)
    H_dim = 2*model.lattice.num_sites_unit_cell
    num_k = len(momentum_path.momenta)
    magnon_Hs_mom_space = np.zeros((num_k, *([H_dim]*order)), dtype=complex)

    for k_idx, k in enumerate(momentum_path.momenta):
        magnon_Hs_mom_space[k_idx] = compute_LSWT_Hamiltonian_momentum_space(
            model, k, magnon_Hs_real_space)

    return magnon_Hs_mom_space


"""
returns: numpy array
    the momentum space BdG Hamiltonian of shape
        [[a_1],
            [],
            [],
            [],
            ...]
"""
def compute_LSWT_Hamiltonian_momentum_space(model: Model, k,
                                            LSWT_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for LSWT of one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and \
        model.lattice.dim != len(k):
        raise Exception(f"dimension of momentum vector " \
                        + f"{len(k)} must equal the "\
                        + f"lattice dimension {model.lattice.dim}")

    order = 2   # LSWT
    magnon_Hs_real_space = LSWT_Hamiltonian_real_space \
        if LSWT_Hamiltonian_real_space is not None \
        else compute_LSWT_Hamiltonian_real_space(model)
    H_dim = 2*model.lattice.num_sites_unit_cell
    sigma_x = np.array([[0, 1], [1, 0]])
    # of the form [a_k^†, a_{-k}] H_BdG [a_k, a_{-k}^†]
    magnon_H_mom_space = np.zeros([H_dim]*order, dtype=complex)

    for coupling in magnon_Hs_real_space:
        subl_idxs = [site.subl_idx for site in coupling.sites]
        bravais_vecs = [
            model.lattice.to_canonical_basis(site.bravais_coords) \
            for site in coupling.sites
        ]

        # need to swap rows to get [a_i^†, a_i] H_BdG [a_j, a_j^†]
        coupling_proper_BdG = coupling.interaction_tensor[::-1]

        phase = np.exp(-1j*np.dot(k, bravais_vecs[0] - bravais_vecs[1]))
        # for +k block
        magnon_H_mom_space[
            2*subl_idxs[0]:2*(subl_idxs[0]+1),
            2*subl_idxs[1]:2*(subl_idxs[1]+1),
        ] += phase * coupling_proper_BdG
        # for -k block
        magnon_H_mom_space.T[
            2*subl_idxs[0]:2*(subl_idxs[0]+1),
            2*subl_idxs[1]:2*(subl_idxs[1]+1),
        ] += np.conj(phase) * sigma_x @ coupling_proper_BdG @ sigma_x
    
    # make BdG Hamiltonians hermitian
    magnon_H_mom_space += np.conj(magnon_H_mom_space.T)
    magnon_H_mom_space /= 2

    return magnon_H_mom_space


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
    eigw_rounded[np.abs(np.real(eigw_rounded)) < 1e-12] = 0
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
        model: Model, k=None, magnon_Hamiltonian=None, orthonormalize=True):
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    bogo_metric = BOGO_METRIC(num_sites_unit_cell)

    H_k = magnon_Hamiltonian if magnon_Hamiltonian is not None \
        else compute_LSWT_Hamiltonian_momentum_space(model, k)

    eigw, eigv = np.linalg.eig(bogo_metric @ H_k)
    # idx = eigw.argsort()
    # eigw, eigv = eigw[idx], eigv[:, idx]
    eigw, eigv = __sort_eigensystem(eigw, eigv)

    if orthonormalize:
        eigv = orthogonalize_wrt_metric(eigv, bogo_metric)
        eigv = normalize_wrt_metric(eigv, bogo_metric)

    return eigw, eigv


def get_eigensystem_along_momentum_path(
        model: Model, momentum_path: ReciprocalLattice.MomentumPath):
    num_ks = len(momentum_path.ks)
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    eigws = np.zeros((num_ks, 2*num_sites_unit_cell))
    eigvs = np.zeros((num_ks, 2*num_sites_unit_cell, 2*num_sites_unit_cell),
        dtype=complex)

    magnon_Hamiltonians = compute_LSWT_Hamiltonian_along_momentum_path(
        model, momentum_path)

    for k_idx, H_k in enumerate(magnon_Hamiltonians):
        eigws[k_idx], eigvs[k_idx] = get_eigensystem_momentum_space(
            model, k=None, magnon_Hamiltonian=H_k)

    return eigws, eigvs



def get_eigensystem_for_Brillouin_zone(model: Model, Nks):
    momenta_BZ = model.lattice.reciprocal_lattice.sample_inverse_unit_cell(Nks)
    return get_eigensystem_for_momentum_meshgrid(momenta_BZ, model)


def get_eigensystem_for_momentum_meshgrid(meshgrid, model: Model):
    Nks = meshgrid.shape[1:]
    dim = meshgrid.shape[0]
    num_samples = np.prod(Nks)
    momenta_BZ = meshgrid.reshape((dim, num_samples)).T
    momentum_path = ReciprocalLattice.MomentumPath(momenta_BZ)

    eigws, eigvs = get_eigensystem_along_momentum_path(model, momentum_path)
    num_bands = eigws.shape[1]
    eigws = eigws.reshape((*Nks, num_bands))
    eigvs = eigvs.reshape((*Nks, num_bands, num_bands))

    return eigws, eigvs


def get_eigensystem_for_loop_momentum(model: Model, ref_momentum, Nks):
    momenta_BZ = model.lattice.reciprocal_lattice.sample_inverse_unit_cell(Nks)
    return __get_eigensystem_for_loop_momentum(model, ref_momentum, momenta_BZ)


def get_eigensystem_for_loop_along_momentum_path(model: Model,
        momentum_path: ReciprocalLattice.MomentumPath, Nks):
    momenta_BZ = model.lattice.reciprocal_lattice.sample_inverse_unit_cell(Nks)
    num_bands = 2 * model.lattice.num_sites_unit_cell
    num_ks_path = len(momentum_path.ks)
    eigws = np.zeros((num_ks_path, *Nks, num_bands))
    eigvs = np.zeros((num_ks_path, *Nks, num_bands, num_bands),
                     dtype=complex)
    for k_idx, k in enumerate(momentum_path.ks):
        eigws[k_idx], eigvs[k_idx] =__get_eigensystem_for_loop_momentum(
            model, k, momenta_BZ)
    return eigws, eigvs


def __get_eigensystem_for_loop_momentum(model: Model, ref_momentum, momenta_BZ):
    Nks = momenta_BZ.shape[:-1]
    ref_momentum = np.expand_dims(ref_momentum, axis=[*range(1, len(Nks)+1)])
    return get_eigensystem_for_momentum_meshgrid(
        ref_momentum - momenta_BZ, model)

    
    
