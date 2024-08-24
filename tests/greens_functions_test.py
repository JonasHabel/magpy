import numpy as np
from magpy.greens_functions import *
from magpy.util import PAULI_MATRICES


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



def test_pole_equation_gradient_descent_non_interacting():
    LSWT_energies = np.array([1., -1., 3., -3.])
    num_bands = len(LSWT_energies) // 2
    reg = 0.01
    close_enough = 1e-3
    def se_and_deriv(omega):
        return np.zeros((num_bands,)*2), np.zeros((num_bands,)*2)

    for init_freq in (0.4, 1.5, 2.3, 4.5, 20.0):
        solution = pole_equation.solve_by_gradient_descent(
            init_freq=init_freq, LSWT_energies=LSWT_energies,
            compute_self_energies_and_derivative_at_freq=se_and_deriv,
            reg=reg, num_steps=100, step_size=0.2, eps=1e-5,
        )
        if solution["converged"]:
            assert np.any(np.abs(solution["freq"] - (LSWT_energies - 1j*reg)) < close_enough)
        else:
            print("not converged")
        



def test_pole_equation_gradient_descent_band_hybridization():
    LSWT_energies = np.array([1., -1., 3., -3.])
    pos_LSWT_energies = LSWT_energies[::2]
    num_bands = len(LSWT_energies) // 2
    V = 0.1
    E_renormalized = np.sum(pos_LSWT_energies)/2 + np.array([1, -1]) * np.sqrt((pos_LSWT_energies[1] - pos_LSWT_energies[0])**2 / 4 + V*np.conj(V))
    reg = 0.01
    close_enough = 1e-3
    def se_and_deriv(omega):
        return np.array([[0, np.conj(V)], [V, 0]]), \
               np.zeros((num_bands,)*2)

    for init_freq in (0.0, 0.4, 1.5, 2.3, 4.5, 20.0):
        solution = pole_equation.solve_by_gradient_descent(
            init_freq=init_freq, LSWT_energies=LSWT_energies,
            compute_self_energies_and_derivative_at_freq=se_and_deriv,
            reg=reg, num_steps=500, step_size=0.1, eps=1e-3,
        )
        if solution["converged"]:
            assert np.any(np.abs(solution["freq"] - (E_renormalized - 1j*reg)) < close_enough)
        else:
            print("not converged")


    def se_and_deriv(omega):
        return np.diag(V*np.conj(V) / (omega - pos_LSWT_energies[::-1] + 1j*reg)), \
               np.diag(V*np.conj(V) / (omega - pos_LSWT_energies[::-1] + 1j*reg)**2),
    
    for init_freq in (0.0, 0.4, 1.5, 2.3, 4.5, 20.0):
        solution = pole_equation.solve_by_gradient_descent(
            init_freq=init_freq, LSWT_energies=LSWT_energies,
            compute_self_energies_and_derivative_at_freq=se_and_deriv,
            reg=reg, num_steps=600, step_size=0.1, eps=1e-4,
        )
        if solution["converged"]:
            assert np.any(np.abs(solution["freq"] - (E_renormalized - 1j*reg)) < close_enough)
        else:
            print("not converged")



def test_pole_equation_gradient_descent_BdG_interaction():
    E = 1.
    V = 0.1
    E_renormalized = np.sqrt(E*E - V*np.conj(V))
    reg = 0.01
    close_enough = 1e-3
    def se_and_deriv(omega):
        return np.array([[V*np.conj(V)/(-omega - E + 1j*reg)]]), \
               np.array([[V*np.conj(V)/(-omega - E + 1j*reg)**2]]),

    for init_freq in (-1.0, 0.0, 1.0, 2.1):
        solution = pole_equation.solve_by_gradient_descent(
            init_freq=init_freq, LSWT_energies=np.array([E, -E]),
            compute_self_energies_and_derivative_at_freq=se_and_deriv,
            reg=reg, num_steps=100, step_size=0.2, eps=1e-5,
        )
        if solution["converged"]:
            assert np.abs(solution["freq"] - (E_renormalized - 1j*reg)) < close_enough
        else:
            print("not converged")