from numba import njit
import numpy as np

@njit
def prod(arr):
    prod = 1
    for x in arr:
        prod *= x
    return prod


@njit
def cum_prod(arr):
    cumprod = np.array(arr)
    for n in range(1, len(arr)):
        cumprod[n] *= cumprod[n-1]
    return cumprod


@njit
def kron_sum(arrs):
    kron_sum = np.zeros(arrs.shape[1] ** arrs.shape[0])
    steps = arrs.shape[1] ** np.arange(arrs.shape[0])
    steps = steps[::-1]
    for idx in np.arange(arrs.shape[0]):
        step = steps[idx]
        for i in range(0, kron_sum.shape[0], arrs.shape[1] * step):
            for j in range(arrs.shape[1]):
                for k in range(j*step, (j+1)*step):
                    kron_sum[i + k] += arrs[idx, j]
    return kron_sum


@njit
def permute(arr, perm):
    out_arr = np.zeros(arr.shape, dtype=arr.dtype)
    for i in range(len(arr)):
        out_arr[i] = arr[perm[i]]
    return out_arr


@njit
def Bose_Einstein(energy, T):
    if T == 0:
        return 0
    return 1.0 / (np.exp(energy/T) - 1)
