import numpy as np
from numba import njit

from ..models import *
from ..greens_functions import get_free_propagator_zero_T
from .util import *
from ..util import PAULI_MATRICES



def compute_one_magnon_self_energy(
        frequencies,
        energies_Gamma,
        eigvs_k,
        cubic_verts,
        eigvs_verts,
        linear_commutator_terms,
        eigvs_commutator_terms,
        num_ks_BZ,
        T, 
        ph_labels,
        reg):
    
    # SETUP
    # 1 for particle, 0 for hole
    ph_idxs = convert_ph_labels_to_indices(ph_labels)
    ph_idxs_vert = \
        np.array([ph_idxs[2][0], ph_idxs[1][0], 1-ph_idxs[0][0]])
    ph_idxs_comm_term = \
        np.array([1-ph_idxs[1][0]])

    num_freqs = len(frequencies)
    num_bands = linear_commutator_terms.shape[-1]
    cubic_vert = cubic_verts[to_binary(ph_idxs_vert)]
    linear_comm_term = linear_commutator_terms[to_binary(ph_idxs_comm_term)]
    pos_energies_Gamma = energies_Gamma[::2]

    # COMPUTE BDG GAUGE PHASES FOR THE CONTRACTED LEGS
    HOLE, PARTICLE = 0, 1
    sigma_x = PAULI_MATRICES[0]
    def conjugate_if(eigvs, condition):
        num_bands = eigvs.shape[-1] // 2
        assert 2*num_bands == eigvs.shape[-2]
        identity = np.eye(num_bands)
        sigma_x = np.kron(identity, PAULI_MATRICES[0])

        return sigma_x @ eigvs.conj() @ sigma_x if condition() else eigvs
    
    eigvs_in = conjugate_if(eigvs_k, lambda: ph_idxs[0][0] == PARTICLE)
    eigvs_out = conjugate_if(eigvs_k, lambda: ph_idxs[2][0] == HOLE)
    gauge_phase_in = compute_gauge_phase(eigvs_in, eigvs_verts[0])[ph_idxs[0][0]::2]
    gauge_phase_internal = compute_gauge_phase(eigvs_verts[1], eigvs_commutator_terms)[ph_idxs[1][0]::2]
    gauge_phase_out = compute_gauge_phase(eigvs_verts[2], eigvs_out)[ph_idxs[2][0]::2]

    # EVALUATE DIAGRAM
    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    
    self_energy[:] = np.einsum(
        "n,mln,l,l,l,m->nm", 
        gauge_phase_in,
        cubic_vert, 
        gauge_phase_internal,
        1.0/(-pos_energies_Gamma + 1j*reg),
        linear_comm_term,
        gauge_phase_out)[np.newaxis]
    
    return self_energy / np.prod(num_ks_BZ)




def compute_gauge_phase(eigvs_1, eigvs_2):
    num_bands = eigvs_1.shape[-1] // 2
    assert 2*num_bands == eigvs_1.shape[-2]
    assert 2*num_bands == eigvs_2.shape[-1]
    assert 2*num_bands == eigvs_2.shape[-2]

    identity = np.eye(num_bands)
    sigma_x = np.kron(identity, PAULI_MATRICES[0])
    sigma_z = np.kron(identity, PAULI_MATRICES[2])
    eigvs_1_inv = sigma_z @ eigvs_1.T.conj() @ sigma_z

    return np.diag(eigvs_1_inv @ sigma_x @ eigvs_2.conj() @ sigma_x)