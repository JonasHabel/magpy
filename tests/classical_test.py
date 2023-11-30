import numpy as np
from magpy import classical, lattice, interactions, models


def test_FM_Heisenberg_square_lattice_total_energy():
    J = -1.0
    latt = lattice.SquareLattice()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=J)
    ]
    mod_2D = models.Model(latt, inter, np.array([[0, 0, 1]]))

    dimensions = (10, 10)
    num_spins = np.prod(dimensions)

    spin_config_FM = np.zeros((*dimensions, 1, 3), dtype=float)
    spin_config_FM[..., 2] = np.array([1])[np.newaxis, np.newaxis, np.newaxis]
    total_energy_FM = classical.compute_total_energy(mod_2D, spin_config_FM)
    expected_total_energy_FM = 2 * J * num_spins
    assert(np.allclose(total_energy_FM, expected_total_energy_FM))
    
    spin_config_Neel = spin_config_FM.copy()
    spin_config_Neel[::2, :] *= -1
    spin_config_Neel[:, ::2] *= -1
    total_energy_Neel = classical.compute_total_energy(mod_2D, spin_config_Neel)
    expected_total_energy_Neel = -2 * J * num_spins
    assert(np.allclose(total_energy_Neel, expected_total_energy_Neel))