import numpy as np
from magpy import time_evolution
from magpy.plot import time_evolution_plot
from magpy.lattice import BravaisLattice, DotLattice
from magpy import models
from magpy.interactions import NthNearestNeighborHeisenbergInteraction
from magpy.largeS import LSWT
from magpy.momenta_utils import Momenta
from . import test_models


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


def test_honeycomb_DMI():
    mod, _ = test_models.FM_Heisenberg_with_DMI_honeycomb()

    lattice_dims = 5, 5
    eigw, eigv = LSWT.get_eigensystems_momentum_space(mod, Momenta.of_BZ(mod.lattice, lattice_dims))
    times = np.linspace(0, 15, 201)

    # init_wavefunction = np.ones((*lattice_dims, 2), dtype=np.complex128)
    # init_wavefunction[lattice_dims[0]//2, lattice_dims[1]//2, 0] = 1.0
    init_wavefunction = time_evolution.get_Gaussian_wave_packet(mod.lattice, lattice_dims, init_pos=np.array([0., 5.]), init_mom=np.array([np.pi/2., 0.]))
    wavefunctions = time_evolution.evolve_exact(mod, times, eigw.raw_quantity[0], eigv.raw_quantity[0], init_wavefunction)

    pos_exp_val = time_evolution.get_expectation_values(wavefunctions, time_evolution.observables.position, mod.lattice)
    # norm = np.linalg.norm(wavefunctions.reshape(len(times), int(np.prod(wavefunctions.shape[1:]))), axis=1)
    # import matplotlib.pyplot as plt
    # plt.plot(np.abs(wavefunctions[:, 4, 4, 0])**2)
    # plt.plot(pos_exp_val[:, 0])
    # plt.plot(pos_exp_val[:, 1])
    # plt.show()

    # time_evolution_plot.plot_time_evolution(wavefunctions, times, mod.lattice, anim_params={"interval": 20})
    assert np.allclose(wavefunctions[0], init_wavefunction)