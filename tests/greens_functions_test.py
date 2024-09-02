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
        for method in (pole_equation.gradient_descent, pole_equation.newton):
            solution = pole_equation.solve(
                init_freq=init_freq, LSWT_energies=LSWT_energies,
                compute_self_energies_and_derivative_at_freq=se_and_deriv,
                reg=reg, num_steps=100, step_size=0.2, eps=1e-5,
                method=method, track_steps=True,
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

    for init_freq in (-20.0, 0.0, 0.4, 1.5, 2.3, 4.5, 20.0):
        for method in (pole_equation.gradient_descent, pole_equation.newton):
            solution = pole_equation.solve(
                init_freq=init_freq, LSWT_energies=LSWT_energies,
                compute_self_energies_and_derivative_at_freq=se_and_deriv,
                reg=reg, num_steps=500, step_size=0.1, eps=1e-3,
                method=method,
            )
            if solution["converged"]:
                assert np.any(np.abs(solution["freq"] - (E_renormalized - 1j*reg)) < close_enough)
            else:
                print("not converged")


    def se_and_deriv(omega):
        return np.diag(V*np.conj(V) / (omega - pos_LSWT_energies[::-1] + 1j*reg)), \
               np.diag(-V*np.conj(V) / (omega - pos_LSWT_energies[::-1] + 1j*reg)**2),
    
    # import matplotlib.pyplot as plt
    # N_real, N_imag = 101, 101
    # omegas_real = np.linspace(0.994, 1.001, N_real)
    # omegas_imag = np.linspace(-0.015, -0.005, N_imag)
    # cost_function = np.zeros((N_real, N_imag))
    # cost_function_deriv = np.zeros((N_real, N_imag), dtype=np.complex128)
    # se = np.zeros((N_real, N_imag, 2, 2), dtype=np.complex128)
    # se_deriv = np.zeros((N_real, N_imag, 2, 2), dtype=np.complex128)
    # for nr, omega_real in enumerate(omegas_real):
    #     for ni, omega_imag in enumerate(omegas_imag):
    #         omega = omega_real + 1j*omega_imag
    #         se[nr, ni], se_deriv[nr, ni] = tuple(se_and_deriv(omega))
    #         cost_function[nr, ni] = pole_equation.cost_function(omega, LSWT_energies[::2], se[nr, ni], reg)
    #         cost_function_deriv[nr, ni] = pole_equation.cost_function_gradient(omega, LSWT_energies[::2], se[nr, ni], se_deriv[nr, ni], reg)
    # solution = pole_equation.solve(
    #     init_freq=0.98, LSWT_energies=LSWT_energies,
    #     compute_self_energies_and_derivative_at_freq=se_and_deriv,
    #     reg=reg, num_steps=200, step_size=0.1, eps=1e-8,
    #     method=pole_equation.gradient_descent,
    #     track_steps=True,
    # )
    # #cost_function_deriv[np.where(np.abs(cost_function_deriv) > 0.005)] = 0
    # cost_function_deriv /= np.abs(cost_function_deriv)
    # plt.contourf(omegas_real, omegas_imag, cost_function.T, levels=np.linspace(0, 0.001, 200))
    # plt.quiver(omegas_real, omegas_imag, np.real(cost_function_deriv.T), -np.imag(cost_function_deriv.T), scale=100)
    # plt.plot([np.real(E_renormalized[1])], [np.imag(E_renormalized[1]) - reg], 'ro')
    # plt.plot(np.real(solution["tracked_quantities"]["freq"]), np.imag(solution["tracked_quantities"]["freq"]), color="red", marker="o")
    # plt.xlim(np.amin(omegas_real), np.amax(omegas_real))
    # plt.ylim(np.amin(omegas_imag), np.amax(omegas_imag))
    # plt.show()
    # plt.contourf(omegas_real, omegas_imag, np.abs(cost_function_deriv).T, levels=np.linspace(-0.05, 0.05, 200), cmap="seismic")
    # plt.plot([np.real(E_renormalized[1])], [np.imag(E_renormalized[1]) - reg], 'ro')
    # plt.plot(np.real(solution["tracked_quantities"]["freq"]), np.imag(solution["tracked_quantities"]["freq"]), color="red", marker="o")
    # plt.xlim(np.amin(omegas_real), np.amax(omegas_real))
    # plt.ylim(np.amin(omegas_imag), np.amax(omegas_imag))
    # plt.show()
    

    for init_freq in (-20.0, 0.0, 0.4, 1.5, 2.3, 4.5, 20.0):
        for method in (pole_equation.gradient_descent, pole_equation.newton):
            solution = pole_equation.solve(
                init_freq=init_freq, LSWT_energies=LSWT_energies,
                compute_self_energies_and_derivative_at_freq=se_and_deriv,
                reg=reg, num_steps=600, step_size=0.1, eps=1e-4,
                method=method,
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

    for init_freq in (-20.0, -1.0, 0.0, 1.0, 2.1, 20.0):
        for method in (pole_equation.gradient_descent, pole_equation.newton):
            solution = pole_equation.solve(
                init_freq=init_freq, LSWT_energies=np.array([E, -E]),
                compute_self_energies_and_derivative_at_freq=se_and_deriv,
                reg=reg, num_steps=100, step_size=0.2, eps=1e-5,
                method=method
            )
            if solution["converged"]:
                assert np.abs(solution["freq"] - (E_renormalized - 1j*reg)) < close_enough
            else:
                print("not converged")