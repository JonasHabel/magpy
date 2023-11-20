import numpy as np
from numba import njit, complex128
from .util_jit import *

@njit
def get_free_propagator_zero_T(omega, energies, ph_signs, reg):
    thermal_factor = 0 if propagator_vanishes_at_zero_T(ph_signs) else 1
    ph_energies = energies.copy()   # signed energies depending on p/h
    for n in range(len(ph_energies)):
        ph_energies[n] *= ph_signs[n]
    return prod(ph_signs) * thermal_factor \
         / (omega - kron_sum(ph_energies) + 1j*reg)


@njit
def get_free_two_magnon_propagator_finite_T(omega, energies, ph_signs, T, reg):
    # TODO this is still wrong. Need to vectorize Bose_Einstein function
    thermal_factor = Bose_Einstein(ph_signs[0]*energies[0], T) \
                   - Bose_Einstein(-ph_signs[1]*energies[1], T)
    return thermal_factor \
         * get_free_propagator_zero_T(omega, energies, ph_signs, reg)


@njit
def get_free_propagator_as_matrix_zero_T(omega, energies, ph_signs, reg):
    return np.diag(get_free_propagator_zero_T(omega, energies, ph_signs, reg))



@njit
def propagator_vanishes_at_zero_T(ph_signs):
    return not np.all(ph_signs == 1) and not np.all(ph_signs == -1)


@njit
def compute_interacting_single_magnon_propagator_at_frequency(
        omega, energies, self_energies, reg):
    return np.linalg.inv(np.diag(omega + 1j*reg - energies) - self_energies)

@njit
def compute_interacting_single_magnon_propagator(
        freqs, energies, self_energies, reg):
    GF = np.zeros(self_energies.shape, dtype=np.complex128)
    for n, omega in enumerate(freqs):
        GF[n] = compute_interacting_single_magnon_propagator_at_frequency(
            omega, energies, self_energies[n], reg)
    return GF
