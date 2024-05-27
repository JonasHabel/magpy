import numpy as np
from numba import njit, complex128
from .util_jit import *

@njit
def get_free_propagator_zero_T(omega, energies, ph_signs, reg):
    thermal_factor = 0 if propagator_vanishes_at_zero_T(ph_signs) else 1
    signed_energies = energies.copy()   # sign is +1/-1 depending on p/h
    for n in range(len(signed_energies)):
        signed_energies[n] *= ph_signs[n]
    return prod(ph_signs) * thermal_factor \
         / (omega - kron_sum(signed_energies) + 1j*reg)


@njit
def get_free_two_magnon_propagator_finite_T(omega, energies, ph_signs, T, reg):
    # TODO this is still wrong. Need to vectorize Bose_Einstein function
    thermal_factor = Bose_Einstein(ph_signs[0]*energies[0], T) \
                   - Bose_Einstein(-ph_signs[1]*energies[1], T)
    return thermal_factor \
         * get_free_propagator_zero_T(omega, energies, ph_signs, reg)


@njit
def get_free_two_magnon_propagator(omega, energies, ph_signs, T, reg):
    signed_energies = energies.copy()   # sign is +1/-1 depending on p/h
    for n in range(len(signed_energies)):
        signed_energies[n] *= ph_signs[n]

    thermal_factors = np.zeros((2, energies.shape[1]))
    thermal_factors[0] = Bose_Einstein_vectorized(signed_energies[0], T)
    thermal_factors[1] = -Bose_Einstein_vectorized(-signed_energies[1], T)
    thermal_factor = kron_sum(thermal_factors)
    
    return -prod(ph_signs) * thermal_factor \
         / (omega - kron_sum(signed_energies) + 1j*reg)


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



@njit
def Bose_Einstein(energy, T):
    if T == 0:
        return -1 if energy < 0 else 0
    return 1.0 / (np.exp(energy/T) - 1)


@njit
def Bose_Einstein_vectorized(energies, T):
    Bose_weights = np.zeros(len(energies), dtype=energies.dtype)
    for n in range(len(energies)):
        Bose_weights[n] = Bose_Einstein(energies[n], T)

    return Bose_weights