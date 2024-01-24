import numpy as np

class Momenta:
    def __init__(self, *k_arrays):
        self.k_arrays = k_arrays
        self.collapsed_k_shape = tuple(np.prod(k_array.shape) for k_array in k_arrays)
        self.restored_k_shape = sum([k_array.shape for k_array in k_arrays], start=tuple())
        self.num_momenta = len(k_arrays)


    def collapse(self, quantity=None, first_momentum_idx=0):
        quantity = self.k_arrays if quantity is None else quantity
        last_momentum_idx = first_momentum_idx + len(self.k_arrays)
        new_shape = (*quantity.shape[:first_momentum_idx], *self.collapsed_k_shape, *quantity.shape[last_momentum_idx:])#

        return quantity.reshape(new_shape)
    

    def restore(self, quantity, first_momentum_idx=0):
        last_momentum_idx = first_momentum_idx + len(self.k_arrays)
        new_shape = (*quantity.shape[:first_momentum_idx], *self.restored_k_shape, *quantity.shape[last_momentum_idx:])

        return quantity.reshape(new_shape)


