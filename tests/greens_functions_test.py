import numpy as np
from ..magpy.greens_functions import *
from ..magpy.util import PAULI_MATRICES


def test_two_band_interacting_greens_functions_single_frequency():
    energies = np.random.rand(2)
    freq = np.random.rand()
    d = np.random.rand(3)   # some random self-energy Bloch vector
    self_energies = np.tensordot(d, PAULI_MATRICES, axes=[[0], [0]])
    
    GF = compute_interacting_single_magnon_propagator_at_frequency(
        freq, energies, self_energies, reg=0.0)
    d_ = d + np.array([0, 0, (energies[0] - energies[1])/2])
    d_sigma = np.tensordot(d_, PAULI_MATRICES, axes=[[0], [0]])
    expected_GF = ((freq - np.sum(energies)/2)*np.eye(2) + d_sigma) \
                / ((freq - np.sum(energies)/2)**2 - d_.dot(d_))
    assert np.allclose(GF, expected_GF)


def test_two_band_interacting_greens_functions_multiple_frequencies():
    energies = np.random.rand(2)
    freqs = np.random.rand(10)
    d = np.random.rand(3)   # some random self-energy Bloch vector
    self_energies = np.tensordot(d, PAULI_MATRICES, axes=[[0], [0]])
    
    GF = compute_interacting_single_magnon_propagator(
        freqs, energies, np.array([self_energies]*len(freqs)), reg=0.0)
    d_ = d + np.array([0, 0, (energies[0] - energies[1])/2])
    d_sigma = np.tensordot(d_, PAULI_MATRICES, axes=[[0], [0]])
    expected_GF = np.array([
        ((freq - np.sum(energies)/2)*np.eye(2) + d_sigma) \
        / ((freq - np.sum(energies)/2)**2 - d_.dot(d_)) \
        for freq in freqs
    ])
    assert np.allclose(GF, expected_GF)
