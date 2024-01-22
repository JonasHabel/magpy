import numpy as np

class Momenta:
    def __init__(self, *k_arrays):
        self.k_arrays = k_arrays
        self.restored_k_shape = sum([k_array.shape for k_array in k_arrays], start=tuple())
        self.num_momenta = len(k_arrays)


    def flatten(self):
        return np.array([
            k_array.reshape(np.prod(k_array.shape)) for k_array in self.k_arrays
        ])
    

    def erect(self, quantity, first_momentum_idx=0):
        last_momentum_idx = first_momentum_idx + len(self.k_arrays)
        new_shape = (*quantity.shape[:first_momentum_idx], *self.restored_k_shape, *quantity.shape[last_momentum_idx:])

        return quantity.reshape(new_shape)


