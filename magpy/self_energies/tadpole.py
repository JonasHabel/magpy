import numpy as np
from numba import njit

from ..models import *
from ..greens_functions import get_free_propagator_zero_T
from .util import *



def compute_one_magnon_self_energy(
        frequencies,
        energies_Gamma,
        cubic_verts,
        linear_commutator_terms,
        T, 
        ph_labels,
        reg):
    
    # 1 for particle, 0 for hole
    ph_idxs = convert_ph_labels_to_indices(ph_labels)
    ph_idxs_vert = \
        np.array([ph_idxs[2][0], ph_idxs[1][0], 1-ph_idxs[0][0]])
    ph_idxs_comm_term = \
        np.array([1-ph_idxs[1][0]])

    num_freqs = len(frequencies)
    num_bands = linear_commutator_terms.shape[-1] // 2
    cubic_vert = cubic_verts[to_binary(ph_idxs_vert)]
    linear_comm_term = linear_commutator_terms[to_binary(ph_idxs_comm_term)]
    pos_energies_Gamma = energies_Gamma[::2]

    self_energy = np.zeros((num_freqs, num_bands, num_bands),
                            dtype=np.complex128)
    
    self_energy = np.einsum(
        "mln,l,l->nm", 
        cubic_vert, pos_energies_Gamma, linear_comm_term)
    
    return self_energy
    