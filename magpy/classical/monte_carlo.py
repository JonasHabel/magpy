import numpy as np
from numba import njit
from magpy.models import Model
from .util import convert_to_flat_index, tensor_contract




class sphere_samplers(object):
    @njit
    def uniform(old_spin=None):
        cos_theta = 2*np.random.rand() - 1
        sin_theta = np.sqrt(1 - cos_theta**2)   # >= 0 since theta \in [0, pi]
        phi = 2*np.pi*np.random.rand()#

        return np.array([
            sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta
        ])


"""
init_spin_config: numpy array of shape (Nx, Ny, ..., #sublattices, 3)
"""
def run_monte_carlo(
        model: Model, num_steps, init_spin_config, temperature, 
        sphere_sampling_func=sphere_samplers.uniform):
    lattice_sizes = init_spin_config.shape[:-2]
    num_unit_cells = int(np.prod(lattice_sizes))
    num_sublattices = model.lattice.num_sites_unit_cell
    num_spins_total = num_unit_cells * num_sublattices

    assert init_spin_config.shape[-2] == num_sublattices
    assert init_spin_config.shape[-1] == 3    # spin vectors should be 3-dimensional

    interactions_by_sublattice = \
        group_interactions_by_sublattice(model.interactions, num_sublattices)
    init_spin_config_flat = init_spin_config.reshape((num_spins_total, 3))

    update_infos = run_monte_carlo_jit(
        interactions_by_sublattice, lattice_sizes, num_sublattices, num_steps, 
        init_spin_config_flat, temperature, sphere_sampling_func
    )

    return update_infos



def reconstruct_spin_config(update_infos, init_spin_config, num_steps=None, intermediate_steps=False):
    if num_steps is None:
        num_steps = len(update_infos[0])    # maximum number of possible steps

    current_spin_config = init_spin_config.copy()   # avoid side effects
    if intermediate_steps:
        intermediate_spin_configs = np.zeros((num_steps+1, *init_spin_config.shape))
        intermediate_spin_configs[0] = init_spin_config

    for n, (accept, bravais_coords, subl_idx, spin) in enumerate(zip(*update_infos)):
        if not accept:
            if intermediate_steps:
                intermediate_spin_configs[n+1] = current_spin_config
            continue
        if n >= num_steps:
            break

        current_spin_config[(*bravais_coords, subl_idx)] = spin
        if intermediate_steps:
            intermediate_spin_configs[n+1] = current_spin_config

    if intermediate_steps:
        return intermediate_spin_configs
    else:
        return current_spin_config



def group_interactions_by_sublattice(interactions, num_sublattices):
    interactions_by_sublattice = [[] for _ in range(num_sublattices)]

    # helper function
    def get_quantity_for_all_sites_except(site_idx, get_quantity, inter):
        return np.array([
            get_quantity(other_site) \
            for other_site_idx, other_site in enumerate(inter.sites) \
            if other_site_idx != site_idx
        ])

    for inter in interactions:
        for site_idx, site in enumerate(inter.sites):
            other_sites_bravais_coords = get_quantity_for_all_sites_except(
                site_idx, lambda other_site: other_site.bravais_coords, inter)
            other_sites_subl_idxs = get_quantity_for_all_sites_except(
                site_idx, lambda other_site: other_site.subl_idx, inter)
            # the transposition makes it easier to compute the 
            # "contracted interactions" in the Metropolis update
            int_tensor_transposed = inter.interaction_tensor.transpose([
                site_idx, 
                *range(site_idx), 
                *range(site_idx+1, len(inter.sites))
            ])
            
            interactions_by_sublattice[site.subl_idx].append([
                other_sites_bravais_coords - site.bravais_coords,
                other_sites_subl_idxs,
                int_tensor_transposed,
            ])

    return interactions_by_sublattice



#@njit
def run_monte_carlo_jit(
        interactions_by_sublattice, lattice_sizes, num_sublattices, num_steps, 
        init_spin_config_flat, temperature, sphere_sampling_func):
    lattice_dim = len(lattice_sizes)
    update_infos = (
        np.zeros((num_steps,), dtype=np.bool),              # has the new spin been accepted?
        np.zeros((num_steps, lattice_dim), dtype=np.int64), # Bravais coords
        np.zeros((num_steps,), dtype=np.int64),             # sublattice idxs
        np.zeros((num_steps, 3), dtype=np.float64),         # the new spin          
    )
    current_spin_config_flat = init_spin_config_flat.copy() # avoid side effects
    # Precompute the norms here already. Avoids recomputing them over and over
    # again in every Metropolis update step
    spin_norms_flat = np.linalg.norm(init_spin_config_flat, axis=-1)

    for n in range(num_steps):
        accepted, new_spin_bravais_coords, new_spin_subl_idx, new_spin_idx_flat, new_spin = \
            Metropolis_update(
                current_spin_config_flat, interactions_by_sublattice, 
                spin_norms_flat, lattice_sizes, num_sublattices, temperature, 
                sphere_sampling_func,
            )
        update_infos[0][n] = accepted
        update_infos[1][n] = new_spin_bravais_coords
        update_infos[2][n] = new_spin_subl_idx
        update_infos[3][n] = new_spin

        # update current configuration accordingly
        if accepted:
            current_spin_config_flat[new_spin_idx_flat] = new_spin

    return update_infos




"""
perform one local Metropolis update step
"""
#@njit
def Metropolis_update(
        current_spin_config_flat, interactions_by_sublattice, spin_norms_flat,
        lattice_sizes, num_sublattices, temperature,
        sphere_sampling_func):
    
    # wiggle a random spin
    lattice_dim = len(lattice_sizes)
    rand_bravais_coords = np.floor(lattice_sizes * np.random.rand(lattice_dim))\
        .astype(np.int64)
    rand_subl_idx = np.floor(num_sublattices * np.random.rand()) \
        .astype(np.int64)
    rand_spin_idx_flat = __convert_to_flat_index(
        rand_bravais_coords, rand_subl_idx, 
        lattice_sizes, num_sublattices
    )
    rand_spin = current_spin_config_flat[rand_spin_idx_flat]
    rand_spin_length = spin_norms_flat[rand_spin_idx_flat]

    wiggled_spin = rand_spin_length * sphere_sampling_func(rand_spin)

    # compute energy gain/loss caused by the wiggling
    contracted_interactions_for_spin = compute_contracted_interactions_for_spin(
        rand_bravais_coords, rand_subl_idx, 
        interactions_by_sublattice, current_spin_config_flat, 
        lattice_sizes, num_sublattices
    )
    energy_last_config = compute_energy_for_spin(
        current_spin_config_flat[rand_spin_idx_flat], 
        contracted_interactions_for_spin
    )
    energy_updated_config = compute_energy_for_spin(
        wiggled_spin, 
        contracted_interactions_for_spin
    )
    energy_diff = energy_updated_config - energy_last_config

    # decide whether to accept updated configuration with the wiggled spin
    accept_probability = min(1.0, np.exp(-energy_diff / temperature))
    accept = np.random.rand() < accept_probability

    return accept, rand_bravais_coords, rand_subl_idx, rand_spin_idx_flat, wiggled_spin
    



#@njit
def compute_contracted_interactions_for_spin(
        spin_bravais_coords, spin_subl_idx,
        interactions_by_sublattice, spin_config_flat, 
        lattice_sizes, num_sublattices):
    interactions_for_spin = interactions_by_sublattice[spin_subl_idx]
    num_interactions_for_spin = len(interactions_for_spin)
    contracted_interactions_for_spin = np.zeros((num_interactions_for_spin, 3))
    
    for ninter, inter in enumerate(interactions_for_spin):
        participating_spins_absolute_bravais_coords = np.array([    # enforce pbc
            (inter_site_relative_bravais_coords + spin_bravais_coords) % np.array(lattice_sizes) \
            for inter_site_relative_bravais_coords in inter[0]
        ], dtype=np.int64)
        participating_spins_subl_idxs = inter[1]
        participating_spins = np.array([
            spin_config_flat[
                __convert_to_flat_index(
                    bravais_coords, subl_idx, lattice_sizes, num_sublattices)
            ] \
            for bravais_coords, subl_idx in zip(
                participating_spins_absolute_bravais_coords,
                participating_spins_subl_idxs
            )
        ])
        int_tensor = inter[2]

        # contracted_interactions_for_spin[ninter] += tensor_contract(
        #     int_tensor, participating_spins
        # )
        einsum_str = "".join([chr(n + 97) for n in range(len(int_tensor.shape))])
        if len(participating_spins) > 0:
            einsum_str += ","
        einsum_str += ",".join([chr(n + 97) for n in range(len(int_tensor.shape) - len(participating_spins), len(int_tensor.shape))])
        contracted_interactions_for_spin[ninter] += np.einsum(
            einsum_str,
            int_tensor, *participating_spins
        )

    return contracted_interactions_for_spin


@njit
def compute_energy_for_spin(spin, contracted_interactions_for_spin):
    energy_for_spin = 0.0
    
    for int_tensor in contracted_interactions_for_spin:
        energy_contribution_by_inter = spin.astype(int_tensor.dtype).dot(int_tensor)
        energy_for_spin += energy_contribution_by_inter

    return energy_for_spin





def __convert_to_flat_index(
        bravais_coords, subl_idx, lattice_sizes, num_sublattices):
    if lattice_sizes == ():
        return subl_idx
    else:
        return convert_to_flat_index(
            bravais_coords, subl_idx, 
            lattice_sizes, num_sublattices)