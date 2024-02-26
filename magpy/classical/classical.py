import numpy as np
from magpy.lattice import BravaisLattice
from magpy.models import Model
from magpy.interactions import Interaction
from magpy import util


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

    lattice_dims = spin_config.shape[:-2]
    bravais_coords = \
        lattice.sample_Bravais_lattice_in_Bravais_coords(lattice_dims)
    
    spin_gradient = np.zeros((2, *spin_config.shape))

    for edge in lattice.edges:
        edge_vector = lattice.get_canonical_coords_for_edge(edge)
        edge_length_sq = np.sum(edge_vector**2)

        for bravais_coord in bravais_coords:
            bravais_coord_pbc = \
                bravais_coord % np.array(lattice_dims)
            other_bravais_coord_pbc = \
                (bravais_coord + edge.bravais_coords) % np.array(lattice_dims)
            spin1 = spin_config[(
                *bravais_coord_pbc, 
                edge.subl_idxs[0]
            )]
            spin2 = spin_config[(
                *other_bravais_coord_pbc, 
                edge.subl_idxs[1]
            )]
            spin_derivative = (spin2 - spin1) / edge_length_sq
            spin_gradient[(slice(None), *bravais_coord_pbc, edge.subl_idxs[0])] += \
                np.outer(edge_vector, spin_derivative)
            spin_gradient[(slice(None), *other_bravais_coord_pbc, edge.subl_idxs[1])] += \
                np.outer(edge_vector, spin_derivative)

    # TODO normalization

    return spin_gradient



def compute_skyrmion_density(lattice: BravaisLattice, spin_config):
    assert lattice.dim == 2

    spin_gradient = compute_spin_gradient(lattice, spin_config)
    skyrmion_density = np.einsum(
        "...a,...a",
        spin_config,
        np.cross(spin_gradient[0], spin_gradient[1]))

    return skyrmion_density