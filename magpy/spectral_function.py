import numpy as np

def compute_spectral_function_at_frequency(greens_function):
    return -1/np.pi * np.imag(np.trace(greens_function))


def compute_spectral_function(greens_functions):
    num_freqs = greens_functions.shape[0]
    spectral_function = np.zeros((num_freqs,), dtype=np.float64)

    for n, greens_function_for_freq in enumerate(greens_functions):
        spectral_function[n] = \
            compute_spectral_function_at_frequency(greens_function_for_freq)

    return spectral_function
