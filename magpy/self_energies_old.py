import numpy as np
from numba import njit
from .models import *
from .greens_functions import get_free_propagator_zero_T, get_free_two_magnon_propagator_finite_T
from .interaction_vertices.util import GET_CUBIC_PERMUTATIONS
from .util import permute



"""
TODO: delete or implement
"""
def compute_one_magnon_one_loop_self_energies_along_momentum_path(
        frequencies,
        energies_along_path, eigvecs_along_path,
        energies_BZ, eigvecs_BZ,
        energies_k_path_minus_BZ, eigvecs_k_path_minus_BZ,
        T, reg):
    num_bands = energies_along_path.shape[1]
    self_energies = np.zeros((2*num_bands), dtype=complex)
    
    

def convert_ph_states_to_indices(particle_hole_states):
    def map_ph_to_idx(ph):
        if ph == "p":
            return 1
        elif ph == "h":
            return 0
        else:
            raise Exception(f"invalid particle-hole state {ph}: "
                          + f"m be either p or h.")
        
    particle_hole_idxs = []
    for ph_state in particle_hole_states:
        particle_hole_idxs.append(list(map(
            map_ph_to_idx, ph_state
        )))

    return particle_hole_idxs



def __get_all_Wick_contractions_of_cubic_vertices_for_loop_momentum(
        cubic_verts_for_loop_momentum, cubic_verts_ph_idxs, N_BZ):
    dim_BZ = len(N_BZ)
    num_bands = cubic_verts_for_loop_momentum.shape[-1] // 2
    CUBIC_PERMUTATIONS = GET_CUBIC_PERMUTATIONS()

    cubic_verts_for_loop_momentum_symmetrized = np.zeros(
        (2, *N_BZ, *([num_bands]*3)),
        dtype=np.complex128)
    
    for vert_idx in range(2):
        conjugate = vert_idx == 1
        for n, permutation in enumerate(CUBIC_PERMUTATIONS):
            ph_idxs = cubic_verts_ph_idxs[vert_idx]
            if conjugate:
                # daggering the vertex means mapping creators <-> annihilators
                # and reversing their order
                ph_idxs = 1 - ph_idxs[::-1]
            ph_idxs_permuted = np.array(permute(ph_idxs, permutation))
            index = (n, ...,
                    *map(lambda ph: slice(ph, None, 2), ph_idxs_permuted))
            # keep BZ momentum indices and permute band indices
            momentum_permutation = [*range(dim_BZ)]
            band_permutation = [x+dim_BZ for x in permutation]
            cubic_verts_for_loop_momentum_permuted = np.transpose(
                cubic_verts_for_loop_momentum[index],
                axes=(momentum_permutation + band_permutation))
            cubic_verts_for_loop_momentum_symmetrized[vert_idx] += \
                cubic_verts_for_loop_momentum_permuted
            
        if conjugate:
            cubic_verts_for_loop_momentum_symmetrized[vert_idx] = \
                np.conj(cubic_verts_for_loop_momentum_symmetrized[vert_idx])

    return cubic_verts_for_loop_momentum_symmetrized


# TODO FIX MOMENTA OF LOOP ENERGIES!
def compute_one_magnon_one_loop_self_energies_at_momentum(
        frequencies,
        energies_BZ,
        energies_k_minus_BZ,
        cubic_verts_for_loop_momentum,
        T, particle_hole_states, reg):
    ph_idxs = convert_ph_states_to_indices(particle_hole_states)
    if len(ph_idxs) != 3:
        raise Exception("One-magnon one-loop self-energy requires one final, " +
                        "one initial, and one intermediate state")
    if len(ph_idxs[0]) != 1 or len(ph_idxs[-1]) != 1:
        raise Exception("Initial and final states must be one-magnon.")
    if any(map(lambda ph: len(ph) != 2, ph_idxs[1:-1])):
        raise Exception("Intermediate state must be two-magnon.")
    
    # 1 for particle, 0 for hole
    cubic_verts_ph_idxs = np.array([
        [ph_idxs[1][0], ph_idxs[1][1], 1-ph_idxs[0][0]],
        [ph_idxs[2][0], 1-ph_idxs[1][0], 1-ph_idxs[1][1]],
    ])
    # 1 for particle, -1 for hole
    intermedate_state_ph_signs = 2*np.array(ph_idxs[1]) - 1

    N_BZ = energies_BZ.shape[:-1]
    # basically Wick-contract every possible combination of vertices
    cubic_verts_for_loop_momentum_symmetrized = \
        __get_all_Wick_contractions_of_cubic_vertices_for_loop_momentum(
            cubic_verts_for_loop_momentum, cubic_verts_ph_idxs, N_BZ)
            
    num_bands = energies_BZ.shape[-1] // 2
    num_ks = int(np.prod(N_BZ))
    cubic_verts_for_loop_momentum_symmetrized_flat = \
        cubic_verts_for_loop_momentum_symmetrized.reshape(
            (2, num_ks, *([num_bands]*3))
        )
    energies_BZ_flat = energies_BZ.reshape(
        (num_ks, 2*num_bands))
    energies_k_minus_BZ_flat = energies_k_minus_BZ.reshape(
        (num_ks, 2*num_bands))
            
    return compute_one_magnon_one_loop_self_energy_at_momentum(
        frequencies, energies_BZ_flat, energies_k_minus_BZ_flat,
        *cubic_verts_for_loop_momentum_symmetrized_flat,
        intermedate_state_ph_signs, T, reg)



def compute_one_magnon_one_loop_self_energy_at_momentum(
        frequencies, energies_BZ_flat, energies_k_minus_BZ_flat,
        cubic_vert_for_loop_momentum_flat1, cubic_vert_for_loop_momentum_flat2,
        ph_signs, T, reg):
    num_freqs = len(frequencies)
    num_bands = energies_BZ_flat.shape[-1] // 2
    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    pos_energies_BZ_flat = energies_BZ_flat[..., ::2]
    pos_energies_k_minus_BZ_flat = energies_k_minus_BZ_flat[..., ::2]
    
    compute_one_magnon_one_loop_self_energy_at_momentum_jit(
        self_energy, frequencies,
        pos_energies_BZ_flat, pos_energies_k_minus_BZ_flat,
        cubic_vert_for_loop_momentum_flat1, cubic_vert_for_loop_momentum_flat2,
        ph_signs, T, reg)
    
    return self_energy


@njit
def compute_one_magnon_one_loop_self_energy_at_momentum_jit(
        out_arr, frequencies,
        pos_energies_BZ_flat, pos_energies_k_minus_BZ_flat,
        cubic_vert_for_loop_momentum_flat1, cubic_verts_for_loop_momentum_flat2,
        ph_signs, T, reg):
    num_ks = pos_energies_BZ_flat.shape[0]
    num_bands = cubic_vert_for_loop_momentum_flat1.shape[-1]

    for nq, pos_energies_at_q, pos_energies_at_k_minus_q, \
        cubic_vert1, cubic_vert2 in zip(
                np.arange(num_ks), pos_energies_BZ_flat,
                pos_energies_k_minus_BZ_flat,
                cubic_vert_for_loop_momentum_flat1,
                cubic_verts_for_loop_momentum_flat2,
            ):
        cubic_vert1_reshaped = np.ascontiguousarray(cubic_vert1) \
            .reshape((num_bands**2, num_bands))
        cubic_vert2_reshaped = np.ascontiguousarray(cubic_vert2) \
            .reshape((num_bands**2, num_bands))
        compute_one_magnon_one_loop_self_energy_at_k_and_q_jit(
            out_arr,
            frequencies, pos_energies_at_q, pos_energies_at_k_minus_q,
            cubic_vert1_reshaped, cubic_vert2_reshaped, ph_signs, T, reg)
        
    loop_factor = 0.5
    out_arr *= loop_factor
    out_arr /= num_ks
    
    

@njit
def compute_one_magnon_one_loop_self_energy_at_k_and_q_jit(
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


