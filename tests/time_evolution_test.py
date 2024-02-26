import numpy as np
from magpy import time_evolution
from magpy.lattice import BravaisLattice, DotLattice
from magpy import models
from magpy.interactions import NthNearestNeighborHeisenbergInteraction
from magpy.largeS import LSWT


def test_quantum_dot():
    latt = DotLattice(2)
    mod = models.Model(latt, [
        NthNearestNeighborHeisenbergInteraction(latt, n=1, J=-1.0)
    ], classical_ground_state=np.array([[0, 0, 1], [0, 0, 1]]))

    eigw, eigv = LSWT.get_eigensystem_momentum_space(mod, k=np.zeros(1))
    times = np.linspace(0, 1, 11)

    init_wavefunction = np.array([1, -1])
    wavefunctions = time_evolution.evolve_exact(mod, times, eigw, eigv, init_wavefunction)
    expected_wavefunctions = np.array([
        init_wavefunction * np.exp(-1j*2*t) for t in times
    ])
    assert np.allclose(wavefunctions, expected_wavefunctions)
    
    init_wavefunction = np.array([1, 1])
    wavefunctions = time_evolution.evolve_exact(mod, times, eigw, eigv, init_wavefunction)
    expected_wavefunctions = np.array([
        init_wavefunction for t in times
    ])
    assert np.allclose(wavefunctions, expected_wavefunctions)
    
    init_wavefunction = np.array([1, 0])
    wavefunctions = time_evolution.evolve_exact(mod, times, eigw, eigv, init_wavefunction)
    expected_wavefunctions = np.array([
        np.array([1 + np.exp(-1j*2*t), 1 - np.exp(-1j*2*t)]) / 2 for t in times
    ])
    assert np.allclose(wavefunctions, expected_wavefunctions)
