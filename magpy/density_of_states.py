import numpy as np
from .lattice import ReciprocalLattice
from .models import Model
from .greens_functions import get_free_propagator_zero_T
from . import LSWT
from . import util_jit
from numba import njit

# def compute_n_magnon_dos_along_momentum_path(
#         n, model, frequencies, momentum_path, energies_along_path, momenta_BZ,
#         energies_BZ, reg):
#     num_ks, num_bands = energies_along_path.shape
#     num_freqs = len(frequencies)
#     Nks_BZ = momenta_BZ.shape[1:]
#     num_ks_BZ = np.prod(Nks_BZ)
#     n_magnon_dos = np.zeros((num_ks, num_freqs))
#     positive_energies_along_path = energies_along_path[:, ::2]
#     positive_energies_BZ_flat = energies_BZ[..., ::2] \
#         .reshape((num_ks_BZ, num_bands//2))

#     for nk, energies_at_k in enumerate(positive_energies_along_path):
#         print(str(nk) + "/" + str(num_ks))
#         n_magnon_dos[nk] = __compute_n_magnon_dos_along_k_path_at(
#             frequencies, model, n, momentum_path.ks[nk],
#             energies_at_k, momenta_BZ, positive_energies_BZ_flat, reg)
            
#     return n_magnon_dos
    


# def __compute_n_magnon_dos_along_k_path_at(frequencies, model, n, k,
#                                            positive_energies_at_k, momenta_BZ,
#                                            positive_energies_BZ_flat, reg):
#     num_freqs = len(frequencies)
#     if n == 1:
#         one_magnon_dos = np.zeros((num_freqs,))
#         for nf, freq in enumerate(frequencies):
#             G0 = get_free_propagator(
#                 freq, np.array([positive_energies_at_k]), reg)
#             one_magnon_dos[nf] = -1/np.pi * np.sum(np.imag(G0))
#         return one_magnon_dos
    
#     elif n == 2:
#         dim = momenta_BZ.shape[0]
#         Nks_BZ = momenta_BZ.shape[1:]
#         num_positive_bands = positive_energies_BZ_flat.shape[-1]
#         num_ks_BZ = np.prod(Nks_BZ)
        
#         k_minus_momenta_BZ = k[(slice(None), *([np.newaxis]*dim))] - momenta_BZ
#         energies_k_minus_BZ = LSWT.get_eigensystem_for_momentum_meshgrid(
#             k_minus_momenta_BZ, model)[0]
#         positive_energies_k_minus_BZ = \
#             energies_k_minus_BZ[..., num_positive_bands:]
#         positive_energies_k_minus_BZ_flat = positive_energies_k_minus_BZ \
#             .reshape((num_ks_BZ, num_positive_bands))
        
#         #momenta_BZ = momenta_BZ.reshape((dim, np.prod(Nks_BZ)))
#         two_magnon_dos = np.zeros((num_freqs,))
#         for nq, energies_at_q, energies_at_k_minus_q in zip(
#                 np.arange(num_ks_BZ),
#                 positive_energies_BZ_flat,
#                 positive_energies_k_minus_BZ_flat):
#             for nf, freq in enumerate(frequencies):
#                 G0 = get_free_propagator(
#                     freq, np.array([energies_at_q, energies_at_k_minus_q]), reg)
#                 two_magnon_dos[nf] += -1/np.pi * np.sum(np.imag(G0))
    
#         return two_magnon_dos / num_ks_BZ
#     else:
#         raise NotImplementedError()




@njit
def compute_n_magnon_dos_along_momentum_path(n, frequencies,
        energies_along_path, energies_BZ, energies_k_path_minus_BZ, reg):
    dim = len(energies_BZ.shape) - 1
    num_ks_path = energies_along_path.shape[0]
    num_bands = energies_along_path.shape[1] // 2
    num_freqs = len(frequencies)
    num_ks_BZ = np.prod(np.array(energies_BZ.shape)[0:dim])
    n_magnon_dos = np.zeros((num_ks_path, num_freqs))
    pos_energies_along_path = energies_along_path[:, ::2]
    pos_energies_BZ_flat = \
        np.ascontiguousarray(energies_BZ[..., ::2]) \
        .reshape((num_ks_BZ, num_bands))
    pos_energies_k_path_minus_BZ_flat = \
        np.ascontiguousarray(energies_k_path_minus_BZ[..., ::2]) \
        .reshape((num_ks_path, num_ks_BZ, num_bands))

    for nk, pos_energies_at_k, pos_energies_k_minus_BZ_flat in zip(
            np.arange(num_ks_path),
            pos_energies_along_path,
            pos_energies_k_path_minus_BZ_flat):
        if nk % 10 == 0:
            print(str(nk) + "/" + str(num_ks_path))
        n_magnon_dos[nk] = __compute_n_magnon_dos_at_k(
            n, frequencies, pos_energies_at_k, pos_energies_BZ_flat,
            pos_energies_k_minus_BZ_flat, reg)
   
    return n_magnon_dos



def compute_n_magnon_dos_at_momentum(n, frequencies,
        energies, energies_BZ, energies_k_minus_BZ, reg):
    dim = len(energies_BZ.shape) - 1
    num_bands = len(energies) // 2
    num_ks_BZ = np.prod(np.array(energies_BZ.shape)[0:dim])
    pos_energies = energies[::2]
    pos_energies_BZ_flat = \
        np.ascontiguousarray(energies_BZ[..., ::2]) \
        .reshape((num_ks_BZ, num_bands))
    pos_energies_k_minus_BZ_flat = \
        np.ascontiguousarray(energies_k_minus_BZ[..., ::2]) \
        .reshape((num_ks_BZ, num_bands))

    n_magnon_dos = __compute_n_magnon_dos_at_k(
        n, frequencies, pos_energies, pos_energies_BZ_flat,
        pos_energies_k_minus_BZ_flat, reg)
   
    return n_magnon_dos


@njit
def __compute_n_magnon_dos_at_k(n, frequencies, pos_energies_at_k,
        pos_energies_BZ_flat, pos_energies_k_minus_BZ_flat, reg):
    num_freqs = len(frequencies)
    if n == 1:
        one_magnon_dos = np.zeros((num_freqs,))
        pos_energies_at_k_reshaped = np.zeros((1, *pos_energies_at_k.shape))
        pos_energies_at_k_reshaped[0] = pos_energies_at_k
        for nf, freq in enumerate(frequencies):
            G0 = get_free_propagator_zero_T(freq, pos_energies_at_k_reshaped,
                                            np.array([1]), reg)
            one_magnon_dos[nf] = -1/np.pi * np.sum(np.imag(G0))
        return one_magnon_dos
    
    elif n == 2:
        num_ks_BZ = pos_energies_BZ_flat.shape[0]
        two_magnon_dos = np.zeros((num_freqs,))
        for nq, energies_at_q, energies_at_k_minus_q in zip(
                np.arange(num_ks_BZ),
                pos_energies_BZ_flat,
                pos_energies_k_minus_BZ_flat):
            pos_energies_loop = np.zeros((2, *energies_at_q.shape))
            pos_energies_loop[0] = energies_at_q
            pos_energies_loop[1] = energies_at_k_minus_q
            for nf, freq in enumerate(frequencies):
                G0 = get_free_propagator_zero_T(
                    freq, pos_energies_loop, np.array([1, 1]), reg)
                two_magnon_dos[nf] += -1/np.pi * np.sum(np.imag(G0))
    
        return two_magnon_dos / num_ks_BZ
    else:
        raise NotImplementedError()
