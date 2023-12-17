import numpy as np
from numba import njit


@njit
def GET_PERMUTATIONS(order: int):
    if order == 3:
        return GET_CUBIC_PERMUTATIONS()
    else:
        raise ValueError("order not supported")

@njit
def GET_CUBIC_PERMUTATIONS():
    return np.array([
        [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0],
    ])

@njit
def GET_INV_CUBIC_PERMUTATIONS():
    return np.array([
        [0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0],
    ])