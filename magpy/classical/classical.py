import numpy as np
from magpy.lattice import BravaisLattice
from magpy.models import Model
from magpy.interactions import Interaction
from magpy import util
from magpy.classical.util import convert_to_flat_index
from numba import njit


def compute_total_energy(model: Model, spin_config):
    dimensions = spin_config.shape[:-2]
    num_sites_unit_cell = model.lattice.num_sites_unit_cell

    assert num_sites_unit_cell == spin_config.shape[-2]

    total_energy = 0.0
    bravais_coords = \
        model.lattice.sample_Bravais_lattice_in_Bravais_coords(dimensions)
    
    for inter in model.interactions:
        int_tensor = inter.interaction_tensor
        relative_bravais_coords_for_inter = np.array([
            site.bravais_coords for site in inter.sites
        ], dtype=int)
        subl_idxs_for_inter = np.array([
            site.subl_idx for site in inter.sites
        ])

        for bravais_coord in bravais_coords:
            absolute_bravais_coords_for_inter = \
                relative_bravais_coords_for_inter + bravais_coord[np.newaxis, :]
            # enforce p.b.c.
            absolute_bravais_coords_for_inter %= np.array(dimensions, dtype=int)
            participating_spins = np.array([
                spin_config[tuple([*abs_bravais_coord, subl_idx])] \
                for abs_bravais_coord, subl_idx in zip(
                    absolute_bravais_coords_for_inter, subl_idxs_for_inter
                )
            ])
            num_participating_spins = len(participating_spins)

            einsum_idxs = \
                util.generate_einsum_indices(range(num_participating_spins))
            einsum_str_int_tensor = "".join(einsum_idxs)
            einsum_str_spins = ",".join(einsum_idxs)
            einsum_str = f"{einsum_str_int_tensor},{einsum_str_spins}"
            
            energy_for_inter = \
                np.einsum(einsum_str, int_tensor, *participating_spins)
            
            total_energy += energy_for_inter

    return total_energy



def compute_spin_gradient(lattice: BravaisLattice, spin_config):
    assert spin_config.shape[-1] == 3

    @njit
    def compute_spin_gradient_for_edge(
            spin_gradient_out_flat, spin_config_flat, 
            edge_bravais_coords, edge_vector, edge_subl_idxs, 
            bravais_coords_lattice, lattice_sizes, num_sublattices):
        edge_length_sq = np.sum(edge_vector**2)
        for bravais_coord in bravais_coords_lattice:
            spin1_bravais_coord_pbc = \
                bravais_coord % lattice_sizes
            spin2_bravais_coord_pbc = \
                (bravais_coord + edge_bravais_coords) % lattice_sizes
            
            spin1_flat_idx = convert_to_flat_index(
                spin1_bravais_coord_pbc, edge_subl_idxs[0], 
                lattice_sizes, num_sublattices)
            spin2_flat_idx = convert_to_flat_index(
                spin2_bravais_coord_pbc, edge_subl_idxs[1], 
                lattice_sizes, num_sublattices)
            spin1 = spin_config_flat[spin1_flat_idx]
            spin2 = spin_config_flat[spin2_flat_idx]
            spin_derivative = (spin2 - spin1) / edge_length_sq

            spin_gradient_out_flat[:, spin1_flat_idx] += \
                np.outer(edge_vector, spin_derivative)
            spin_gradient_out_flat[:, spin2_flat_idx] += \
                np.outer(edge_vector, spin_derivative)
            
            
    lattice_sizes = spin_config.shape[:-2]
    num_unit_cells = int(np.prod(lattice_sizes))
    num_sublattices = lattice.num_sites_unit_cell
    num_sites_total = num_unit_cells * num_sublattices
    
    spin_config_flat = spin_config.reshape((num_sites_total, 3))
    bravais_coords_lattice = \
        lattice.sample_Bravais_lattice_in_Bravais_coords(lattice_sizes)
    
    spin_gradient_flat = np.zeros((2, *spin_config_flat.shape))

    for edge in lattice.edges:
        edge_vector = lattice.get_canonical_coords_for_edge(edge)
        compute_spin_gradient_for_edge(
            spin_gradient_flat, spin_config_flat, 
            edge.bravais_coords, edge_vector, edge.subl_idxs, 
            bravais_coords_lattice, np.array(lattice_sizes), num_sublattices)
        

    # TODO normalization
        
    spin_gradient = spin_gradient_flat.reshape((2, *spin_config.shape))

    return spin_gradient



def compute_skyrmion_density(lattice: BravaisLattice, spin_config):
    assert lattice.dim == 2

    spin_gradient = compute_spin_gradient(lattice, spin_config)
    skyrmion_density = np.einsum(
        "...a,...a",
        spin_config,
        np.cross(spin_gradient[0], spin_gradient[1]))

    return skyrmion_density