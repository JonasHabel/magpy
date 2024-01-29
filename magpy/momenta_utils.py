import numpy as np
from functools import wraps


class Momenta:
    def __init__(self, *k_arrays):
        self.k_arrays = k_arrays
        self.collapsed_shapes = tuple(int(np.prod(k_array.shape[:-1])) for k_array in k_arrays)
        self.collapsed_tensor_shape_deep = (int(np.prod(np.array(self.collapsed_shapes))), )
        self.restored_shapes = tuple(k_array.shape[:-1] for k_array in k_arrays)
        self.restored_tensor_shape = tuple(dim for shape in self.restored_shapes for dim in shape)
        self.num_momenta = len(k_arrays)


    def collapse(self, quantities_arr=None, first_momentum_idx=0):
        quantities_arr = self.k_arrays if quantities_arr is None \
                else quantities_arr.k_arrays if isinstance(quantities_arr, Momenta) \
                else quantities_arr
        collapsed_quantities = []
        for quantity_arr, collapsed_shape, restored_shape in \
                zip(quantities_arr, self.collapsed_shapes, self.restored_shapes):
            last_momentum_idx = first_momentum_idx + len(restored_shape)
            new_shape = (*quantity_arr.shape[:first_momentum_idx], collapsed_shape, *quantity_arr.shape[last_momentum_idx:])
            collapsed_quantities.append(quantity_arr.reshape(new_shape))

        return collapsed_quantities
    

    def collapse_tensor(self, quantity, first_momentum_idx=0, deep=False):
        last_momentum_idx = first_momentum_idx + len(self.restored_tensor_shape)
        new_shape = (
            *quantity.shape[:first_momentum_idx], 
            *(self.collapsed_tensor_shape_deep if deep else self.collapsed_shapes),
            *quantity.shape[last_momentum_idx:])
        return quantity.reshape(new_shape)
    

    def restore(self, quantities, first_momentum_idx=0):
        quantities = self.k_arrays if quantities is None \
                else quantities.k_arrays if isinstance(quantities, Momenta) \
                else quantities
        restored_quantities = []
        for quantity_arr, collapsed_shape, restored_shape in \
                zip(quantities, self.collapsed_shapes, self.restored_shapes):
            last_momentum_idx = first_momentum_idx + 1
            new_shape = (*quantity_arr.shape[:first_momentum_idx], *restored_shape, *quantity_arr.shape[last_momentum_idx:])
            restored_quantities.append(quantity_arr.reshape(new_shape))

        return restored_quantities
    

    def restore_tensor(self, quantity, first_momentum_idx=0, deep=False):
        last_momentum_idx = first_momentum_idx + (1 if deep else len(self.collapsed_shapes))
        new_shape = (*quantity.shape[:first_momentum_idx], *self.restored_tensor_shape, *quantity.shape[last_momentum_idx:])
        return quantity.reshape(new_shape)



class Target:
    def __init__(self, arg_idx, first_momentum_idx, is_tensor=False, collapse_deep=False):
        self.arg_idx = arg_idx
        self.first_momentum_idx = first_momentum_idx
        self.is_tensor = is_tensor
        self.collapse_deep = collapse_deep


def CollapseMomenta(
        momentum_arrays_arg_idx=0, 
        targets=(Target(arg_idx=1, first_momentum_idx=0, is_tensor=False, collapse_deep=False))):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            momentum_arrays = args[momentum_arrays_arg_idx]
            collapsed_args = [*args]
            if isinstance(momentum_arrays, Momenta):
                for target in targets:
                    if target.is_tensor:
                        collapsed_args[target.arg_idx] = \
                            momentum_arrays.collapse_tensor(
                                args[target.arg_idx], 
                                target.first_momentum_idx, 
                                target.collapse_deep,
                            )
                    else:
                        collapsed_args[target.arg_idx] = \
                            momentum_arrays.collapse(
                                args[target.arg_idx], 
                                target.first_momentum_idx, 
                            )

            result = func(*collapsed_args, **kwargs)

            return result
        
        return wrapped_func
    
    return decorator


def RestoreMomenta(
        momentum_arrays_arg_idx=0,
        output_first_momentum_idx=0,
        output_is_tensor=True,
        output_restore_deep=False,
        custom_restore_func=None):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            momentum_arrays = args[momentum_arrays_arg_idx]

            result = func(*args, **kwargs)

            if isinstance(momentum_arrays, Momenta):
                if custom_restore_func is None:
                    if output_is_tensor:
                        result = momentum_arrays.restore_tensor(result, output_first_momentum_idx, output_restore_deep)
                    else:
                        result = momentum_arrays.restore(result, output_first_momentum_idx)
                else:
                    result = custom_restore_func(result, momentum_arrays)

            return result
        
        return wrapped_func
    
    return decorator