import numpy as np
from ..magpy.greens_functions import *
from ..magpy.spectral_function import *
from ..magpy.util import PAULI_MATRICES


def test_single_band_non_interacting_spectral_function_single_frequency():
    energy = np.random.rand()
    freq = np.random.rand()
    reg = 0.1
    self_energies = 0.0
    
    GF = compute_interacting_single_magnon_propagator_at_frequency(
        freq, np.array([energy]), np.array([[self_energies]]), reg)
    SF = compute_spectral_function_at_frequency(GF)
    expected_SF = 1/np.pi * reg / ((freq - energy)**2 + reg**2)

    assert np.allclose(SF, expected_SF)


def test_two_band_non_interacting_spectral_function_single_frequency():
    energies = np.random.rand(2)
    freq = np.random.rand()
    reg = 0.1
    self_energies = np.zeros((2, 2), dtype=complex)
    
    GF = compute_interacting_single_magnon_propagator_at_frequency(
        freq, energies, self_energies, reg)
    SF = compute_spectral_function_at_frequency(GF)

    expected_SF = 1/np.pi * np.sum(reg / ((freq - energies)**2 + reg**2))
    assert np.allclose(SF, expected_SF)


def test_two_band_non_interacting_spectral_function_multiple_frequencies():
    energies = np.array([0.5, 1.6])
    freqs = np.linspace(0, 2, 20)
    reg = 0.1
    self_energies = np.zeros((len(freqs), 2, 2), dtype=complex)
    
    GFs = compute_interacting_single_magnon_propagator(
        freqs, energies, self_energies, reg)
    SF = compute_spectral_function(GFs)

    expected_SF = 1/np.pi * (reg / ((freqs - energies[0])**2 + reg**2) \
                           + reg / ((freqs - energies[1])**2 + reg**2))
    # import matplotlib.pyplot as plt
    # plt.scatter(freqs, SF)
    # plt.show()
    assert np.allclose(SF, expected_SF)