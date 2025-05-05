import numpy as np
from numba import njit

from ..models import *
from ..greens_functions import get_free_two_magnon_propagator
from .util import *
from ..util_jit import factorial


def compute_full_one_magnon_self_energy(
    frequencies,
    energies_BZ,
    energies_minus_k_minus_BZ,
    cubic_verts,
    T, 
    ph_labels_intermediate_state,
    reg,
    freq_derivatives=None,
    two_magnon_propagator=get_free_two_magnon_propagator,
):
    # 1 for particle, 0 for hole
    ph_idxs_intermediate_state = convert_ph_labels_to_indices(
        [ph_labels_intermediate_state]
    )[0]
    
    ANNIHILATOR, CREATOR = 0, 1
    cubic_vert = np.zeros(
        (*cubic_verts.shape[1:-1], 2*cubic_verts.shape[-1]), 
        dtype=np.complex128,
    )
    cubic_vert[..., 0::2] = \
        cubic_verts[to_binary([*ph_idxs_intermediate_state, ANNIHILATOR])]
    cubic_vert[..., 1::2] = \
        cubic_verts[to_binary([*ph_idxs_intermediate_state, CREATOR])]
    cubic_vert_left = cubic_vert
    cubic_vert_right = cubic_vert.conj()

    self_energy = compute_one_magnon_self_energy_from_vertices_without_prefactor(
        frequencies,
        energies_BZ,
        energies_minus_k_minus_BZ,
        cubic_vert_left, cubic_vert_right,
        T, 
        ph_idxs_intermediate_state,
        reg,
        freq_derivatives,
        two_magnon_propagator,
    )

    # apply diagram prefactor for each particle/hole sector separately
    for ph_idx_in in (ANNIHILATOR, CREATOR):
        ph_idxs_left_vert = \
            np.array([*ph_idxs_intermediate_state, ph_idx_in])
        
        for ph_idx_out in (ANNIHILATOR, CREATOR):
            ph_idxs_right_vert = \
                np.array([*ph_idxs_intermediate_state, ph_idx_out])
            
            diagram_prefactor = compute_diagram_prefactor(
                    ph_idxs_verts=[ph_idxs_left_vert, ph_idxs_right_vert],
                    ph_idxs_loops=[ph_idxs_intermediate_state],
                    num_internal_propagators=2)
            
            self_energy[..., ph_idx_in::2, ph_idx_out::2] *= diagram_prefactor


    return self_energy if freq_derivatives else self_energy[0]



def compute_one_magnon_self_energy(
    frequencies,
    energies_BZ,
    energies_minus_k_minus_BZ,
    cubic_verts,
    T, 
    ph_labels,
    reg,
    freq_derivatives=None,
    two_magnon_propagator=get_free_two_magnon_propagator,
):
    # 1 for particle, 0 for hole
    ph_idxs = convert_ph_labels_to_indices(ph_labels)
    ph_idxs_left_vert = \
        np.array([ph_idxs[1][1], ph_idxs[1][0], 1-ph_idxs[0][0]])
    ph_idxs_right_vert = \
        np.array([ph_idxs[1][1], ph_idxs[1][0], 1-ph_idxs[2][0]])
    ph_idxs_intermediate_state = ph_idxs[1]
    
    cubic_vert_left = cubic_verts[to_binary(ph_idxs_left_vert)]
    cubic_vert_right = cubic_verts[to_binary(ph_idxs_right_vert)] \
        .conj()
    
    self_energy = compute_one_magnon_self_energy_from_vertices_without_prefactor(
        frequencies,
        energies_BZ,
        energies_minus_k_minus_BZ,
        cubic_vert_left, cubic_vert_right,
        T, 
        ph_idxs_intermediate_state,
        reg,
        freq_derivatives,
        two_magnon_propagator,
    )
    
    self_energy *= compute_diagram_prefactor(
        ph_idxs_verts=[ph_idxs_left_vert, ph_idxs_right_vert],
        ph_idxs_loops=[ph_idxs[1]],
        num_internal_propagators=2)
    
    return self_energy if freq_derivatives else self_energy[0]



def compute_one_magnon_self_energy_from_vertices_without_prefactor(
    frequencies,
    energies_BZ,
    energies_minus_k_minus_BZ,
    cubic_vert_left, cubic_vert_right,
    T, 
    ph_idxs_intermediate_state,
    reg,
    freq_derivatives=None,
    two_magnon_propagator=get_free_two_magnon_propagator,
):
    # 1 for particle, -1 for hole
    intermedate_state_ph_signs = 2*np.array(ph_idxs_intermediate_state) - 1

    N_BZ = energies_BZ.shape[:-1]
    num_freqs = len(frequencies)
    num_ks_BZ = int(np.prod(N_BZ))
    num_bands = energies_BZ.shape[-1] // 2

    assert cubic_vert_left.shape[:-1] == (*N_BZ, num_bands, num_bands)
    assert cubic_vert_right.shape[:-1] == (*N_BZ, num_bands, num_bands)

    dim_in_leg = cubic_vert_left.shape[-1]
    dim_out_leg = cubic_vert_right.shape[-1]

    cubic_vert_left_flat = cubic_vert_left \
        .reshape((num_ks_BZ, num_bands**2, dim_in_leg))
    cubic_vert_right_flat = cubic_vert_right \
        .reshape((num_ks_BZ, num_bands**2, dim_out_leg))
    pos_energies_BZ_flat = energies_BZ[..., ::2] \
        .reshape((num_ks_BZ, num_bands))
    pos_energies_minus_k_minus_BZ_flat = energies_minus_k_minus_BZ[..., ::2] \
        .reshape((num_ks_BZ, num_bands))
    
    num_derivatives = len(freq_derivatives) if freq_derivatives else 1
    self_energy = np.zeros((num_derivatives, num_freqs, dim_in_leg, dim_out_leg),
                            dtype=np.complex128)
    
    compute_one_magnon_self_energy_jit(
        self_energy, frequencies,
        pos_energies_BZ_flat, pos_energies_minus_k_minus_BZ_flat,
        cubic_vert_left_flat, cubic_vert_right_flat,
        intermedate_state_ph_signs, T, reg,
        freq_derivatives if freq_derivatives else (0,),
        two_magnon_propagator,
    )
    
    self_energy /= num_ks_BZ
        
    return self_energy
    


@njit
def compute_one_magnon_self_energy_jit(
    out_arr, frequencies,
    pos_energies_BZ_flat, pos_energies_minus_k_minus_BZ_flat,
    cubic_vert_flat1, cubic_vert_flat2,
    ph_signs, T, reg, freq_derivatives, two_magnon_propagator,
):
    num_ks = pos_energies_BZ_flat.shape[0]
    num_bands = cubic_vert_flat1.shape[-1]

    for nq, pos_energies_at_q, pos_energies_at_minus_k_minus_q, \
        cubic_vert1, cubic_vert2 in zip(
                np.arange(num_ks), pos_energies_BZ_flat,
                pos_energies_minus_k_minus_BZ_flat,
                cubic_vert_flat1,
                cubic_vert_flat2,
            ):
        cubic_vert1 = np.ascontiguousarray(cubic_vert1)
        cubic_vert2 = np.ascontiguousarray(cubic_vert2)
        compute_one_magnon_self_energy_loop_integral_contribution_jit(
            out_arr,
            frequencies, pos_energies_at_q, pos_energies_at_minus_k_minus_q,
            cubic_vert1, cubic_vert2, ph_signs, T, reg, freq_derivatives,
            two_magnon_propagator,
        )
    
    

@njit
def compute_one_magnon_self_energy_loop_integral_contribution_jit(
    out_arr,
    frequencies, pos_energies_at_q, pos_energies_at_minus_k_minus_q,
    cubic_vert1, cubic_vert2, ph_signs, T, reg, freq_derivatives,
    two_magnon_propagator,
):
    pos_energies_in_propagator = np.zeros((2, len(pos_energies_at_q)))
    pos_energies_in_propagator[0] = pos_energies_at_minus_k_minus_q
    pos_energies_in_propagator[1] = pos_energies_at_q

    for nf, freq in enumerate(frequencies):
        for nderiv, deriv_order in enumerate(freq_derivatives):
            G0_deriv = two_magnon_propagator(
                freq, pos_energies_in_propagator, ph_signs, T, reg, deriv_order)
            out_arr[nderiv, nf] += (cubic_vert1.T * G0_deriv) @ cubic_vert2


