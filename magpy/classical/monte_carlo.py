import numpy as np
from numba import njit
from numba.typed import List
from magpy.models import Model
from .util import convert_to_flat_index, tensor_contract_jit




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
    lattice_sizes = np.array(init_spin_config.shape[:-2])
    if model.lattice.dim == 0: # workaround for numba, which cannot deal well with empty tuples
        lattice_sizes = np.array([1])
    num_unit_cells = int(np.prod(lattice_sizes))
    num_sublattices = model.lattice.num_sites_unit_cell
    num_spins_total = num_unit_cells * num_sublattices

    assert init_spin_config.shape[-2] == num_sublattices
    assert init_spin_config.shape[-1] == 3    # spin vectors should be 3-dimensional

    interactions_by_sublattice = \
        group_interactions_by_sublattice(
            model.interactions, model.lattice.dim, num_sublattices)
    init_spin_config_flat = init_spin_config \
        .reshape((num_spins_total, 3)) \
        .astype(np.float64)

    update_infos, final_spin_config_flat = run_monte_carlo_jit(
        interactions_by_sublattice, lattice_sizes, num_sublattices, num_steps, 
        init_spin_config_flat, temperature, sphere_sampling_func
    )

    final_spin_config = \
        final_spin_config_flat.reshape((*lattice_sizes, num_sublattices, 3))
    if model.lattice.dim == 0:
        update_infos[1] = np.zeros((0, model.lattice.embedding_dim))    # reset Bravais coords after workaround for numba

    return update_infos, final_spin_config



def reconstruct_spin_config(update_infos, init_spin_config, num_steps=None, intermediate_steps=False):
    if num_steps is None or num_steps > len(update_infos[0]):
        num_steps = len(update_infos[0])    # maximum number of possible steps

    current_spin_config = init_spin_config.copy()   # avoid side effects
    if intermediate_steps:
        intermediate_spin_configs = np.zeros((num_steps+1, *init_spin_config.shape))
        intermediate_spin_configs[0] = init_spin_config

    for n, (accept, bravais_coords, subl_idx, spin) in enumerate(zip(*update_infos)):
        if n >= num_steps:
            break
        if not accept:
            if intermediate_steps:
                intermediate_spin_configs[n+1] = current_spin_config
            continue

        current_spin_config[(*bravais_coords, subl_idx)] = spin
        if intermediate_steps:
            intermediate_spin_configs[n+1] = current_spin_config

    if intermediate_steps:
        return intermediate_spin_configs
    else:
        return current_spin_config


def get_accepted_updates(update_infos, with_acceptance_ratio=False):
    accept, bravais_coords, subl_idxs, spins = update_infos
    if any(map(lambda x: len(x) <= 0, update_infos)):
        return update_infos
    
    lattice_dim = len(bravais_coords[0])
    accepted_update_idxs = np.where(accept == True)[0]
    num_accepted_updates = len(accepted_update_idxs)
    accepted_update_infos = (
        np.zeros((num_accepted_updates, lattice_dim), dtype=np.int64), # Bravais coords
        np.zeros((num_accepted_updates,), dtype=np.int64),             # sublattice idxs
        np.zeros((num_accepted_updates, 3), dtype=np.float64),         # the new spin        
    )

    for quantity_idx in range(3):
        for n, accepted_update_idx in enumerate(accepted_update_idxs):
            accepted_update_infos[quantity_idx][n] = \
                update_infos[quantity_idx+1][accepted_update_idx]
            
    if with_acceptance_ratio:
        num_total_updates = len(accept)
        acceptance_ratio = num_accepted_updates / num_total_updates
        return accepted_update_infos, acceptance_ratio
    else:
        return accepted_update_infos



MAGIC_NUMBER = 1.23456789e-30

def group_interactions_by_sublattice(interactions, lattice_dim, num_sublattices):
    interactions_by_sublattice = [List([]) for _ in range(num_sublattices)]

    # helper function
    def get_quantity_for_all_sites_except(site_idx, get_quantity, inter):
        return np.array([
            get_quantity(other_site) \
            for other_site_idx, other_site in enumerate(inter.sites) \
            if other_site_idx != site_idx
        ])

    for inter in interactions:
        num_sites = len(inter.sites)
        num_other_sites = num_sites - 1
        for site_idx, site in enumerate(inter.sites):
            other_sites_bravais_coords = get_quantity_for_all_sites_except(
                site_idx, lambda other_site: other_site.bravais_coords - site.bravais_coords, inter)
            other_sites_subl_idxs = get_quantity_for_all_sites_except(
                site_idx, lambda other_site: other_site.subl_idx, inter)
            # the transposition makes it easier to compute the 
            # "contracted interactions" in the Metropolis update
            int_tensor_transposed = inter.interaction_tensor.transpose([
                site_idx, 
                *range(site_idx), 
                *range(site_idx+1, num_sites)
            ])
            
            # need to compress the information into a 1d float numpy array (in order to have a homogeneous list).
            # This array first contains all bravais coordinates of the participating spins chained together;
            # next, all sublattice indices of the participating spins chained together;
            # a separator number, the MAGIC_NUMBER (very hacky!);
            # and finally, the interaction tensor as a flattened 1d array.
            # This is super ugly but the only way I found that numba can deal with it
            interaction_compressed = np.zeros((
                lattice_dim*num_other_sites + num_other_sites + 1 + 3**num_sites
            ), dtype=np.float64)
            interaction_compressed[:lattice_dim*num_other_sites] = \
                other_sites_bravais_coords.reshape((lattice_dim*num_other_sites,))
            interaction_compressed[lattice_dim*num_other_sites:(lattice_dim+1)*num_other_sites] = \
                other_sites_subl_idxs
              # separator between coordinates and interaction tensor; hacky hack!!
            interaction_compressed[(lattice_dim+1)*num_other_sites] = MAGIC_NUMBER
            interaction_compressed[(lattice_dim+1)*num_other_sites+1:] = \
                int_tensor_transposed.reshape((3**num_sites,))
            interactions_by_sublattice[site.subl_idx].append(interaction_compressed)

    return List(interactions_by_sublattice)



@njit
def run_monte_carlo_jit(
        interactions_by_sublattice, lattice_sizes, num_sublattices, num_steps, 
        init_spin_config_flat, temperature, sphere_sampling_func):
    lattice_dim = len(lattice_sizes)
    update_infos = (
        np.zeros((num_steps,), dtype=np.bool_),             # has the new spin been accepted?
        np.zeros((num_steps, lattice_dim), dtype=np.int64), # Bravais coords
        np.zeros((num_steps,), dtype=np.int64),             # sublattice idxs
        np.zeros((num_steps, 3), dtype=np.float64),         # the new spin          
    )
    current_spin_config_flat = init_spin_config_flat.copy() # avoid side effects
    # Precompute the norms here already. Avoids recomputing them over and over
    # again in every Metropolis update step
    spin_norms_flat = np.zeros((len(init_spin_config_flat)))
    for n in range(len(spin_norms_flat)):
        spin_norms_flat[n] = np.linalg.norm(init_spin_config_flat[n])
    # spin_norms_flat = np.linalg.norm(init_spin_config_flat, axis=-1)

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

    return update_infos, current_spin_config_flat




"""
perform one local Metropolis update step
"""
@njit
def Metropolis_update(
        current_spin_config_flat, interactions_by_sublattice, spin_norms_flat,
        lattice_sizes, num_sublattices, temperature,
        sphere_sampling_func):
    
    # wiggle a random spin
    lattice_dim = len(lattice_sizes)
    rand_bravais_coords = np.floor(lattice_sizes * np.random.rand(lattice_dim))\
        .astype(np.int64)
    rand_subl_idx = int(np.floor(num_sublattices * np.random.rand()))
    rand_spin_idx_flat = convert_to_flat_index(
        rand_bravais_coords, rand_subl_idx, 
        lattice_sizes, num_sublattices
    )
    rand_spin = current_spin_config_flat[rand_spin_idx_flat]
    rand_spin_length = spin_norms_flat[rand_spin_idx_flat]

    wiggled_spin = rand_spin_length * sphere_sampling_func(rand_spin)

    # compute energy gain/loss caused by the wiggling
    contracted_interactions_for_spin = compute_contracted_interactions_for_spin(
        rand_bravais_coords, 
        interactions_by_sublattice[rand_subl_idx], current_spin_config_flat, 
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
    



@njit
def compute_contracted_interactions_for_spin(
        spin_bravais_coords,
        interactions_for_spin, spin_config_flat, 
        lattice_sizes, num_sublattices):
    #interactions_for_spin = interactions_by_sublattice[spin_subl_idx]
    lattice_dim = len(spin_bravais_coords)
    num_interactions_for_spin = len(interactions_for_spin)
    contracted_interactions_for_spin = np.zeros((num_interactions_for_spin, 3))
    
    for ninter, inter in enumerate(interactions_for_spin):
        # unpack flattened data
        num_participating_spins = np.where(inter == MAGIC_NUMBER)[0][0] // (lattice_dim + 1)
        participating_spins_relative_bravais_coords = inter[:num_participating_spins*lattice_dim].reshape((num_participating_spins, lattice_dim))
        participating_spins_subl_idxs = inter[num_participating_spins*lattice_dim:num_participating_spins*(lattice_dim+1)]
        int_tensor_flat = inter[num_participating_spins*(lattice_dim+1)+1:]

        participating_spins = np.zeros((num_participating_spins, 3))
        for n in range(num_participating_spins):
            # enforce pbc
            participating_spin_absolute_bravais_coords = \
                (participating_spins_relative_bravais_coords[n] + spin_bravais_coords) % lattice_sizes
            participating_spin_subl_idx = int(participating_spins_subl_idxs[n])
            
            participating_spins[n] = spin_config_flat[
                convert_to_flat_index(
                    participating_spin_absolute_bravais_coords.astype(np.int64),
                    participating_spin_subl_idx, 
                    lattice_sizes, num_sublattices)
            ]

        contracted_interactions_for_spin[ninter] += tensor_contract_jit(
            int_tensor_flat, participating_spins, first_arg_is_flat=True
        )
        # einsum_str = "".join([chr(n + 97) for n in range(len(int_tensor.shape))])
        # if len(participating_spins) > 0:
        #     einsum_str += ","
        # einsum_str += ",".join([chr(n + 97) for n in range(len(int_tensor.shape) - len(participating_spins), len(int_tensor.shape))])
        # contracted_interactions_for_spin[ninter] += np.einsum(
        #     einsum_str,
        #     int_tensor, *participating_spins
        # )

    return contracted_interactions_for_spin


@njit
def compute_energy_for_spin(spin, contracted_interactions_for_spin):
    energy_for_spin = 0.0
    
    for int_tensor in contracted_interactions_for_spin:
        energy_contribution_by_inter = \
            spin.astype(int_tensor.dtype)\
                .dot(np.ascontiguousarray(int_tensor))
        energy_for_spin += energy_contribution_by_inter

    return energy_for_spin





def __convert_to_flat_index(
        bravais_coords, subl_idx, lattice_sizes, num_sublattices):
    # if lattice_sizes == ():
    #     return subl_idx
    # else:
    return convert_to_flat_index(
        bravais_coords, subl_idx, 
        lattice_sizes, num_sublattices)