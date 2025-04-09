import numpy as np
from numba import njit

from ..models import *
from ..greens_functions import get_free_propagator_zero_T
from .util import *
from ..util import PAULI_MATRICES


def compute_full_one_magnon_self_energy(
    frequencies,
    eigvs_k,
    quadratic_commutator_terms,
    eigvs_commutator_terms,
    num_ks_BZ,
    T,
):
    # SETUP
    num_freqs = len(frequencies)
    num_bands = quadratic_commutator_terms.shape[-1]

    ANNIHILATOR, CREATOR = 0, 1
    quadratic_comm_term = np.zeros(
        (*quadratic_commutator_terms.shape[1:-2], 2*num_bands, 2*num_bands), 
        dtype=np.complex128,
    )
    quadratic_comm_term[..., 0::2, 0::2] = \
        quadratic_commutator_terms[to_binary([CREATOR, ANNIHILATOR])]
    quadratic_comm_term[..., 1::2, 0::2] = \
        quadratic_commutator_terms[to_binary([ANNIHILATOR, ANNIHILATOR])]
    quadratic_comm_term[..., 0::2, 1::2] = \
        quadratic_commutator_terms[to_binary([CREATOR, CREATOR])]
    quadratic_comm_term[..., 1::2, 1::2] = \
        quadratic_commutator_terms[to_binary([ANNIHILATOR, CREATOR])]
    
    # COMPUTE BDG GAUGE PHASES FOR THE CONTRACTED LEGS
    eigvs_in = BdG_conjugate(eigvs_k)
    eigvs_out = eigvs_k
    sigma_x = np.kron(np.eye(num_bands), PAULI_MATRICES[0])
    gauge_phase_in = compute_gauge_phase(eigvs_commutator_terms[1], eigvs_in)
    gauge_phase_out = sigma_x @ compute_gauge_phase(eigvs_commutator_terms[0], eigvs_out) @ sigma_x
    
    # EVALUATE DIAGRAM
    self_energy = np.zeros((num_freqs, 2*num_bands, 2*num_bands),
                            dtype=np.complex128)
    
    self_energy[:] = np.einsum(
        "Nn,MN,Mm->nm", 
        gauge_phase_in,
        quadratic_comm_term, 
        gauge_phase_out)[np.newaxis]
    
    
    # apply diagram prefactor for each particle/hole sector separately
    for ph_idx_in in (ANNIHILATOR, CREATOR):
        for ph_idx_out in (ANNIHILATOR, CREATOR):
            ph_idxs_comm_term = np.array([ph_idx_out, 1-ph_idx_in])
            diagram_prefactor = compute_diagram_prefactor(
                    ph_idxs_verts=[ph_idxs_comm_term],
                    ph_idxs_loops=[],
                    num_internal_propagators=0)
            
            self_energy[..., ph_idx_in::2, ph_idx_out::2] *= diagram_prefactor
    
    return self_energy / np.prod(num_ks_BZ)



def compute_one_magnon_self_energy(
    frequencies,
    eigvs_k,
    quadratic_commutator_terms,
    eigvs_commutator_terms,
    num_ks_BZ,
    T, 
    ph_labels
):
    np.set_printoptions(suppress=True, precision=3) 
    # SETUP
    # 1 for particle, 0 for hole
    ph_idxs = convert_ph_labels_to_indices(ph_labels)
    ph_idxs_comm_term = \
        np.array([ph_idxs[1][0], 1-ph_idxs[0][0]])
    
    num_freqs = len(frequencies)
    num_bands = quadratic_commutator_terms.shape[-1]
    quadratic_comm_term = quadratic_commutator_terms[to_binary(ph_idxs_comm_term)]
    
    # COMPUTE BDG GAUGE PHASES FOR THE CONTRACTED LEGS
    HOLE, PARTICLE = 0, 1
    
    eigvs_in = BdG_conjugate(eigvs_k) #BdG_conjugate_if(eigvs_k, lambda: ph_idxs[0][0] == PARTICLE)
    eigvs_out = eigvs_k #BdG_conjugate_if(eigvs_k, lambda: ph_idxs[1][0] == HOLE)
    gauge_phase_in = compute_gauge_phase(eigvs_commutator_terms[1], eigvs_in)[1-ph_idxs[0][0]::2, 1-ph_idxs[0][0]::2]
    gauge_phase_out = compute_gauge_phase(eigvs_commutator_terms[0], eigvs_out)[ph_idxs[1][0]::2, ph_idxs[1][0]::2]
    
    # EVALUATE DIAGRAM
    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    
    self_energy[:] = np.einsum(
        "Nn,MN,Mm->nm", 
        gauge_phase_in,
        quadratic_comm_term, 
        gauge_phase_out)[np.newaxis]
    
    
    self_energy *= compute_diagram_prefactor(
        ph_idxs_verts=[ph_idxs_comm_term],
        ph_idxs_loops=[],
        num_internal_propagators=0,
    )
    
    return self_energy / np.prod(num_ks_BZ)


