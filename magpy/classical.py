import numpy as np
from .models import Model
from .interactions import Interaction
from . import util


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


