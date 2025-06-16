import numpy as np
from numba import njit

from ..models import *
from ..greens_functions import get_free_propagator_zero_T
from .util import *
from ..util import PAULI_MATRICES



def _internal_one_magnon_propagator(pos_energies_Gamma):
    return 1.0 / (-pos_energies_Gamma)


def compute_full_one_magnon_self_energy(
    frequencies,
    energies_Gamma,
    eigvs_k,
    cubic_verts,
    eigvs_verts,
    linear_commutator_terms,
    eigvs_commutator_terms,
    num_ks_BZ,
    T, 
    ph_labels_intermediate_state,
    reg,
    internal_one_magnon_propagator=_internal_one_magnon_propagator,
):
    # 1 for particle, 0 for hole
    ph_idxs_intermediate_state = np.array(convert_ph_labels_to_indices(
        [ph_labels_intermediate_state]
    )[0])
    ph_idxs_comm_term = 1 - ph_idxs_intermediate_state

    num_freqs = len(frequencies)
    num_bands = linear_commutator_terms.shape[-1]

    ANNIHILATOR, CREATOR = 0, 1
    cubic_vert = np.zeros(
        (*cubic_verts.shape[1:-3], 2*num_bands, num_bands, 2*num_bands), 
        dtype=np.complex128,
    )
    cubic_vert[..., 0::2, :, 0::2] = \
        cubic_verts[to_binary([CREATOR, ph_idxs_intermediate_state[0], ANNIHILATOR])]   # -k-q, q, k
    cubic_vert[..., 1::2, :, 0::2] = \
        cubic_verts[to_binary([ANNIHILATOR, ph_idxs_intermediate_state[0], ANNIHILATOR])]
    cubic_vert[..., 0::2, :, 1::2] = \
        cubic_verts[to_binary([CREATOR, ph_idxs_intermediate_state[0], CREATOR])]
    cubic_vert[..., 1::2, :, 1::2] = \
        cubic_verts[to_binary([ANNIHILATOR, ph_idxs_intermediate_state[0], CREATOR])]
    linear_comm_term = linear_commutator_terms[to_binary(ph_idxs_comm_term)]
    pos_energies_Gamma = energies_Gamma[::2]

    # COMPUTE BDG GAUGE PHASES FOR THE CONTRACTED LEGS
    
    eigvs_in = BdG_conjugate(eigvs_k)
    eigvs_out = eigvs_k
    sigma_x = np.kron(np.eye(num_bands), PAULI_MATRICES[0])
    gauge_phase_in = compute_gauge_phase(eigvs_verts[2], eigvs_in)
    gauge_phase_internal = compute_gauge_phase(eigvs_verts[1], eigvs_commutator_terms)[ph_idxs_intermediate_state[0]::2, ph_idxs_intermediate_state[0]::2]
    gauge_phase_out = sigma_x @ compute_gauge_phase(eigvs_verts[0], eigvs_out) @ sigma_x

    # EVALUATE DIAGRAM
    self_energy = np.zeros((num_freqs, 2*num_bands, 2*num_bands),
                            dtype=np.complex128)
    
    self_energy[:] = np.einsum(
        "Nn,MLN,Ll,l,l,Mm->nm", 
        gauge_phase_in,
        cubic_vert, 
        gauge_phase_internal,
        internal_one_magnon_propagator(pos_energies_Gamma),
        linear_comm_term,
        gauge_phase_out)[np.newaxis]
    
    # apply diagram prefactor for each particle/hole sector separately
    HOLE, PARTICLE = 0, 1
    for ph_idx_in in (HOLE, PARTICLE):
        for ph_idx_out in (HOLE, PARTICLE):
            ph_idxs_vert = \
                np.array([ph_idx_out, ph_idxs_intermediate_state[0], 1-ph_idx_in])
            diagram_prefactor = compute_diagram_prefactor(
                    ph_idxs_verts=[ph_idxs_vert, ph_idxs_comm_term],
                    ph_idxs_loops=[],
                    num_internal_propagators=1)
            
            self_energy[..., 1-ph_idx_in::2, 1-ph_idx_out::2] *= diagram_prefactor

    # self_energy *= compute_diagram_prefactor(
    #     ph_idxs_verts=[ph_idxs_vert, ph_idxs_comm_term],
    #     ph_idxs_loops=[],
    #     num_internal_propagators=1,
    # )
    
    return self_energy / np.prod(num_ks_BZ)



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
    reg,
    internal_one_magnon_propagator=_internal_one_magnon_propagator,
):
    
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
    
    eigvs_in = BdG_conjugate(eigvs_k) #BdG_conjugate_if(eigvs_k, lambda: ph_idxs[0][0] == PARTICLE)
    eigvs_out = eigvs_k #BdG_conjugate_if(eigvs_k, lambda: ph_idxs[2][0] == HOLE)
    gauge_phase_in = compute_gauge_phase(eigvs_verts[2], eigvs_in)[1-ph_idxs[0][0]::2, 1-ph_idxs[0][0]::2]
    gauge_phase_internal = compute_gauge_phase(eigvs_verts[1], eigvs_commutator_terms)[ph_idxs[1][0]::2, ph_idxs[1][0]::2]
    gauge_phase_out = compute_gauge_phase(eigvs_verts[0], eigvs_out)[ph_idxs[2][0]::2, ph_idxs[2][0]::2]

    # EVALUATE DIAGRAM
    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    
    self_energy[:] = np.einsum(
        "Nn,MLN,Ll,l,l,Mm->nm", 
        gauge_phase_in,
        cubic_vert, 
        gauge_phase_internal,
        internal_one_magnon_propagator(pos_energies_Gamma),
        linear_comm_term,
        gauge_phase_out)[np.newaxis]
    

    self_energy *= compute_diagram_prefactor(
        ph_idxs_verts=[ph_idxs_vert, ph_idxs_comm_term],
        ph_idxs_loops=[],
        num_internal_propagators=1,
    )
    
    return self_energy / np.prod(num_ks_BZ)

