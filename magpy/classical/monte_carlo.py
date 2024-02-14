import numpy as np
from numba import njit
from magpy.models import Model
from .util import convert_to_flat_index, tensor_contract



"""
init_spin_config: numpy array of shape (Nx, Ny, ..., #sublattices, 3)
"""
def run_monte_carlo(model: Model, num_steps, init_spin_config, temperature):
    lattice_sizes = init_spin_config.shape[:-2]
    num_unit_cells = int(np.prod(lattice_sizes))
    num_sublattices = model.lattice.num_sites_unit_cell
    num_spins_total = num_unit_cells * num_sublattices

    assert init_spin_config[-2] == num_sublattices
    assert init_spin_config[-1] == 3    # spin vectors should be 3-dimensional

    interactions_by_sublattice = \
        group_interactions_by_sublattice(model.interactions, num_sublattices)
    init_spin_config_flat = init_spin_config.reshape((num_spins_total, 3))

    return run_monte_carlo_jit(
        interactions_by_sublattice, lattice_sizes, num_sublattices, num_steps, 
        init_spin_config_flat, temperature
    )



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
        init_spin_config_flat, temperature):
    num_spins_total = np.prod(lattice_sizes) * num_sublattices
    spin_configs_flat = np.zeros(
        (num_steps+1, num_spins_total, 3), 
        dtype=np.float64
    )
    spin_configs_flat[0] = init_spin_config_flat

    for n in range(1, num_steps+1):
        spin_configs_flat[n] = Metropolis_update(
            spin_configs_flat[n-1], interactions_by_sublattice,
            lattice_sizes, num_sublattices, temperature,
        )

    return spin_configs_flat


"""
perform one local Metropolis update step
"""
#@njit
def Metropolis_update(
        prev_spin_config_flat, interactions_by_sublattice, 
        lattice_sizes, num_sublattices, temperature):
    
    # wiggle a random spin
    lattice_dim = len(lattice_sizes)
    rand_bravais_coords = np.floor(lattice_sizes * np.random.rand(lattice_dim))\
        .astype(np.int64)
    rand_subl_idx = np.floor(num_sublattices * np.random.rand()) \
        .astype(np.int64)
    rand_spin_idx_flat = convert_to_flat_index(
        rand_bravais_coords, rand_subl_idx, 
        lattice_sizes, num_sublattices
    )
    rand_spin = prev_spin_config_flat[rand_spin_idx_flat]

    wiggled_spin = sample_sphere_uniform(radius=np.linalg.norm(rand_spin))

    # compute energy gain/loss caused by the wiggling
    contracted_interactions_for_spin = compute_contracted_interactions_for_spin(
        rand_bravais_coords, rand_subl_idx, 
        interactions_by_sublattice, prev_spin_config_flat, 
        lattice_sizes, num_sublattices
    )
    energy_last_config = compute_energy_for_spin(
        prev_spin_config_flat[rand_spin_idx_flat], 
        contracted_interactions_for_spin
    )
    energy_updated_config = compute_energy_for_spin(
        wiggled_spin, 
        contracted_interactions_for_spin
    )
    energy_diff = energy_updated_config - energy_last_config

    # decide whether to accept updated configuration with the wiggled spin
    next_spin_config_flat = prev_spin_config_flat.copy() # avoid side effects
    accept_probability = min(1.0, np.exp(-energy_diff / temperature))
    if np.random.rand() < accept_probability:
        next_spin_config_flat[rand_spin_idx_flat] = wiggled_spin

    return next_spin_config_flat
    


@njit
def sample_sphere_uniform(radius):
    cos_theta = 2*np.random.rand() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)   # >= 0 since theta \in [0, pi]
    phi = 2*np.pi*np.random.rand()

    return radius * np.array([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta])


#@njit
def compute_contracted_interactions_for_spin(
        spin_bravais_coords, spin_subl_idx,
        interactions_by_sublattice, spin_config_flat, 
        lattice_sizes, num_sublattices):
    interactions_for_spin = interactions_by_sublattice[spin_subl_idx]
    num_interactions_for_spin = len(interactions_for_spin)
    contracted_interactions_for_spin = np.zeros((num_interactions_for_spin, 3))
    
    for ninter, inter in enumerate(interactions_for_spin):
        participating_spins_absolute_bravais_coords = np.array([
            inter_site_relative_bravais_coords + spin_bravais_coords \
            for inter_site_relative_bravais_coords in inter[0]
        ], dtype=np.int64)
        participating_spins_subl_idxs = inter[1]
        participating_spins = np.array([
            spin_config_flat[
                convert_to_flat_index(
                    bravais_coords, subl_idx, lattice_sizes, num_sublattices)
            ] \
            for bravais_coords, subl_idx in zip(
                participating_spins_absolute_bravais_coords,
                participating_spins_subl_idxs
            )
        ])
        int_tensor = inter[2]

        contracted_interactions_for_spin[ninter] += tensor_contract(
            int_tensor, participating_spins
        )

    return contracted_interactions_for_spin


@njit
def compute_energy_for_spin(spin, contracted_interactions_for_spin):
    energy_for_spin = 0.0
    
    for int_tensor in contracted_interactions_for_spin:
        energy_contribution_by_inter = spin.astype(int_tensor.dtype).dot(int_tensor)
        energy_for_spin += energy_contribution_by_inter

    return energy_for_spin





