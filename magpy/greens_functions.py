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
def get_free_two_magnon_propagator(
        omega, energies, ph_signs, T, reg, freq_derivative_order=0):
    signed_energies = energies.copy()   # sign is +1/-1 depending on p/h
    for n in range(len(signed_energies)):
        signed_energies[n] *= ph_signs[n]
    E = kron_sum(signed_energies)

    thermal_factors = np.zeros((2, energies.shape[1]))
    thermal_factors[0] = Bose_Einstein_vectorized(signed_energies[0], T)
    thermal_factors[1] = -Bose_Einstein_vectorized(-signed_energies[1], T)
    thermal_factor = kron_sum(thermal_factors)
    
    derivative_prefactor = \
        (-1)**freq_derivative_order * factorial(freq_derivative_order)

    return -prod(ph_signs) * thermal_factor * derivative_prefactor \
        / (omega - E + 1j*reg)**(freq_derivative_order+1)


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
    EPS = 1e-12

    if T == 0:
        return -1 if energy < EPS else 0
    return 1.0 / (np.exp(energy/T) - 1)


@njit
def Bose_Einstein_vectorized(energies, T):
    Bose_weights = np.zeros(len(energies), dtype=energies.dtype)
    for n in range(len(energies)):
        Bose_weights[n] = Bose_Einstein(energies[n], T)

    return Bose_weights




class pole_equation:

    def solve_by_gradient_descent(
        init_freq, LSWT_energies, compute_self_energies_and_derivative_at_freq,
        reg, num_steps=10, step_size=0.1, eps=1e-3,
    ):
        omega = init_freq
        eps_squared = eps * eps

        for n in range(num_steps):
            self_energies, self_energies_derivative = \
                compute_self_energies_and_derivative_at_freq(omega)
            cost_func, precomputed_values = \
                pole_equation.cost_function(
                    omega, LSWT_energies, self_energies, reg,
                    return_precomputed_values=True,
                )
            
            if cost_func < eps_squared:
                return {"freq": omega, "converged": True, "num_steps": n, "error": cost_func}
            
            cost_func_gradient = pole_equation.cost_function_gradient(
                omega, LSWT_energies, self_energies, self_energies_derivative,
                reg, precomputed_values,
            )
            omega -= step_size * 2*np.conj(cost_func_gradient)
                
        return {"freq": omega, "converged": False, "num_steps": num_steps, "error": cost_func}



    # returns the cost function f(w) = |det G^{-1}(w)|^2
    # which is to be minimized.
    def cost_function(
        omega, LSWT_energies, self_energies, reg,
        return_precomputed_values=False,
    ):
        G_inv = np.diag(omega + 1j*reg - LSWT_energies) - self_energies
        det_G_inv = np.linalg.det(G_inv)
        cost_func = det_G_inv * np.conj(det_G_inv)

        if return_precomputed_values:
            return cost_func, {"G_inv": G_inv, "cost_func": cost_func}
        
        return cost_func


    # Returns the gradient of the cost function f(w):
    #    df/dw = |det G^{-1}(w)|^2 * Tr(G(w)(1 - dΣ*/dw))
    def cost_function_gradient(
        omega, LSWT_energies, self_energies, self_energies_derivative, reg,
        precomputed_values=None,
    ):
        # def get_or_else(key, dct, func):
        #     if dct is not None and key in dct:
        #         return dct[key]
        #     return func()
        
        # G_inv = get_or_else(
        #     "G_inv", precomputed_values,
        #     lambda: np.diag(omega + 1j*reg - LSWT_energies) - self_energies,
        # )
        G_inv = np.diag(omega + 1j*reg - LSWT_energies) - self_energies
        det_G_inv = np.linalg.det(G_inv)
        
        G = np.linalg.inv(G_inv)
        identity = np.eye(len(LSWT_energies))

        # cost_func = get_or_else(
        #     "cost_func", precomputed_values,
        #     lambda: pole_equation.cost_function(
        #         omega, LSWT_energies, self_energies, reg),
        cost_func = det_G_inv * np.conj(det_G_inv)
        cost_func_gradient = cost_func * np.trace(G @ (identity - self_energies_derivative))

        return cost_func_gradient
        
    
