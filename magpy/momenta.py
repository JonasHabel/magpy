import numpy as np
from functools import wraps


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



def CollapseMomenta(
        momentum_arrays_arg_idx=0, 
        target_arg_idxs=(1,), 
        target_first_momentum_idxs=(0,), 
        output_first_momentum_idx=0):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            momentum_arrays = args[momentum_arrays_arg_idx]
            collapsed_args = [*args]
            if isinstance(momentum_arrays, Momenta):
                for target_arg_idx, target_first_momentum_idx in (target_arg_idxs, target_first_momentum_idxs):
                    collapsed_args[target_arg_idx] = momentum_arrays.collapse(
                        args[target_arg_idx], target_first_momentum_idx)

            result = func(*collapsed_args, **kwargs)

            if isinstance(momentum_arrays, Momenta):
                result = momentum_arrays.restore(result, output_first_momentum_idx)

            return result
        
        return wrapped_func
    
    return decorator