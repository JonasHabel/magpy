import numpy as np
from numba import njit

from .models import *
from .greens_functions import get_free_propagator_zero_T


def __convert_ph_labels_to_indices(particle_hole_labels):
    def map_label_to_idx(ph):
        if ph == "p":
            return 1
        elif ph == "h":
            return 0
        else:
            raise Exception(f"invalid particle-hole state {ph}: "
                          + f"m be either p or h.")
        
    particle_hole_idxs = []
    for ph_label in particle_hole_labels:
        particle_hole_idxs.append(list(map(
            map_label_to_idx, ph_label
        )))

    return particle_hole_idxs



def __to_binary(bits):
    return sum(2**i * bit for i, bit in enumerate(reversed(bits)))



def compute_one_magnon_self_energy_bubble(
        frequencies,
        energies_BZ,
        energies_k_minus_BZ,
        cubic_verts,
        T, 
        ph_labels,
        reg):
    
    # 1 for particle, 0 for hole
    ph_idxs = __convert_ph_labels_to_indices(ph_labels)
    ph_idxs_left_vert = \
        np.array([ph_idxs[1][1], ph_idxs[1][0], 1-ph_idxs[0][0]])
    ph_idxs_right_vert = \
        np.array([ph_idxs[1][1], ph_idxs[1][0], 1-ph_idxs[2][0]])
    # 1 for particle, -1 for hole
    intermedate_state_ph_signs = 2*np.array(ph_idxs[1]) - 1

    N_BZ = energies_BZ.shape[:-1]
    num_freqs = len(frequencies)
    num_ks = int(np.prod(N_BZ))
    num_bands = energies_BZ.shape[-1] // 2
    cubic_vert_left_flat = cubic_verts[__to_binary(ph_idxs_left_vert)] \
        .reshape((num_ks, *([num_bands]*3)))
    cubic_vert_right_flat = cubic_verts[__to_binary(ph_idxs_right_vert)] \
        .reshape((num_ks, *([num_bands]*3))) \
        .conj()
    pos_energies_BZ_flat = energies_BZ[..., ::2].reshape(
        (num_ks, num_bands))
    pos_energies_k_minus_BZ_flat = energies_k_minus_BZ[..., ::2].reshape(
        (num_ks, num_bands))
    
    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    
    compute_one_magnon_self_energy_bubble_jit(
        self_energy, frequencies,
        pos_energies_BZ_flat, pos_energies_k_minus_BZ_flat,
        cubic_vert_left_flat, cubic_vert_right_flat,
        intermedate_state_ph_signs, T, reg)
    
    return self_energy
    


@njit
def compute_one_magnon_self_energy_bubble_jit(
        out_arr, frequencies,
        pos_energies_BZ_flat, pos_energies_k_minus_BZ_flat,
        cubic_vert_flat1, cubic_vert_flat2,
        ph_signs, T, reg):
    num_ks = pos_energies_BZ_flat.shape[0]
    num_bands = cubic_vert_flat1.shape[-1]

    for nq, pos_energies_at_q, pos_energies_at_k_minus_q, \
        cubic_vert1, cubic_vert2 in zip(
                np.arange(num_ks), pos_energies_BZ_flat,
                pos_energies_k_minus_BZ_flat,
                cubic_vert_flat1,
                cubic_vert_flat2,
            ):
        cubic_vert1_reshaped = np.ascontiguousarray(cubic_vert1) \
            .reshape((num_bands**2, num_bands))
        cubic_vert2_reshaped = np.ascontiguousarray(cubic_vert2) \
            .reshape((num_bands**2, num_bands))
        compute_one_magnon_self_energy_bubble_loop_integral_contribution_jit(
            out_arr,
            frequencies, pos_energies_at_q, pos_energies_at_k_minus_q,
            cubic_vert1_reshaped, cubic_vert2_reshaped, ph_signs, T, reg)
        
    num_Wick_contractions = 2.0 # TODO
    out_arr *= num_Wick_contractions
    out_arr /= num_ks
    
    

@njit
def compute_one_magnon_self_energy_bubble_loop_integral_contribution_jit(
        out_arr,
        frequencies, pos_energies_at_q, pos_energies_at_k_minus_q,
        cubic_vert1, cubic_vert2, ph_signs, T, reg):
    pos_energies_in_propagator = np.zeros((2, len(pos_energies_at_q)))
    pos_energies_in_propagator[0] = pos_energies_at_q
    pos_energies_in_propagator[1] = pos_energies_at_k_minus_q

    for nf, freq in enumerate(frequencies):
        G0 = get_free_propagator_zero_T(
            freq, pos_energies_in_propagator, ph_signs, reg)
        out_arr[nf] += (cubic_vert1.T * G0) @ cubic_vert2


