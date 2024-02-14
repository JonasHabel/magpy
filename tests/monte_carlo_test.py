import numpy as np
from magpy import lattice, interactions, models
from magpy.classical import monte_carlo, util
from . import test_models


def test_monte_carlo_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()
    np.random.seed(1)

    lattice_sizes = (10,)

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    #spin_config[..., 2] = np.array([S])[np.newaxis, np.newaxis, np.newaxis]
    spin_config[:, 0, :] = S * np.array([
        [np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, lattice_sizes[0], endpoint=False)
    ])

    interactions_by_sublattice = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.num_sites_unit_cell)
    assert len(interactions_by_sublattice) == model.lattice.num_sites_unit_cell
    assert len(interactions_by_sublattice[0]) == 2
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice[0]))
    bravais_coords_0, subl_idxs_0, int_tensor_0 = interactions_by_sublattice[0][0]
    assert np.allclose(bravais_coords_0, np.array([[1]]))
    assert np.allclose(subl_idxs_0, np.array([0]))
    assert np.allclose(int_tensor_0, J*np.eye(3))
    bravais_coords_1, subl_idxs_1, int_tensor_1 = interactions_by_sublattice[0][1]
    assert np.allclose(bravais_coords_1, np.array([[-1]]))
    assert np.allclose(subl_idxs_1, np.array([0]))
    assert np.allclose(int_tensor_1, J*np.eye(3))

    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4]), 0,
        interactions_by_sublattice, spin_config.reshape((lattice_sizes[0], 3)), 
        lattice_sizes, model.lattice.num_sites_unit_cell)
    assert np.allclose(contracted_interactions_for_spin, np.array([
        J*spin_config[5, 0], J*spin_config[3, 0]
    ]))

    energy_for_spin_1 = monte_carlo.compute_energy_for_spin(
        spin_config[4, 0], contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_1, 2*J * S**2 * np.cos(2*np.pi/10))
    energy_for_spin_2 = monte_carlo.compute_energy_for_spin(
        np.array([S, 0, 0]), contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_2, J * S**2 * (np.cos(2*np.pi*3/10) + np.cos(2*np.pi*5/10)))

    # to make sure the function runs without runtime errors
    monte_carlo.Metropolis_update(
        spin_config.reshape((lattice_sizes[0], 3)), interactions_by_sublattice, 
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J))




def test_monte_carlo_AFM_Heisenberg_chain():
    B = np.array([0, 0, 1])
    model, (J, S_A, S_B) = test_models.AFM_Heisenberg_chain(B)
    np.random.seed(1)

    lattice_sizes = (10,)

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    spin_config[:, 0, :] = S_A * np.array([
        [np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, lattice_sizes[0], endpoint=False)
    ])
    spin_config[:, 1 , 2] = np.array([S_B])[np.newaxis, np.newaxis, np.newaxis]

    interactions_by_sublattice = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.num_sites_unit_cell)
    assert len(interactions_by_sublattice) == model.lattice.num_sites_unit_cell
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice))
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice[0]))
    bravais_coords_A0, subl_idxs_A0, int_tensor_A0 = interactions_by_sublattice[0][0]
    assert np.allclose(bravais_coords_A0, np.array([[0]]))
    assert np.allclose(subl_idxs_A0, np.array([1]))
    assert np.allclose(int_tensor_A0, J*np.eye(3))
    bravais_coords_A1, subl_idxs_A1, int_tensor_A1 = interactions_by_sublattice[0][1]
    assert np.allclose(bravais_coords_A1, np.array([[-1]]))
    assert np.allclose(subl_idxs_A1, np.array([1]))
    assert np.allclose(int_tensor_A1, J*np.eye(3))
    bravais_coords_A2, subl_idxs_A2, int_tensor_A2 = interactions_by_sublattice[0][2]
    assert np.allclose(bravais_coords_A2, np.array([]))
    assert np.allclose(subl_idxs_A2, np.array([]))
    assert np.allclose(int_tensor_A2, -B)
    #
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice[1]))
    bravais_coords_B0, subl_idxs_B0, int_tensor_B0 = interactions_by_sublattice[1][0]
    assert np.allclose(bravais_coords_B0, np.array([[0]]))
    assert np.allclose(subl_idxs_B0, np.array([0]))
    assert np.allclose(int_tensor_B0, J*np.eye(3))
    bravais_coords_B1, subl_idxs_B1, int_tensor_B1 = interactions_by_sublattice[1][1]
    assert np.allclose(bravais_coords_B1, np.array([[1]]))
    assert np.allclose(subl_idxs_B1, np.array([0]))
    assert np.allclose(int_tensor_B1, J*np.eye(3))
    bravais_coords_B2, subl_idxs_B2, int_tensor_B2 = interactions_by_sublattice[1][2]
    assert np.allclose(bravais_coords_B2, np.array([]))
    assert np.allclose(subl_idxs_B2, np.array([]))
    assert np.allclose(int_tensor_B2, -B)

    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4]), 1,
        interactions_by_sublattice, spin_config.reshape((2*lattice_sizes[0], 3)), 
        lattice_sizes, model.lattice.num_sites_unit_cell)
    assert np.allclose(contracted_interactions_for_spin, np.array([
        J*spin_config[4, 0], J*spin_config[5, 0], -B
    ]))

    energy_for_spin_1 = monte_carlo.compute_energy_for_spin(
        spin_config[4, 1], contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_1, J * S_A*S_B * (np.sin(2*np.pi*4/10) + np.sin(2*np.pi*5/10)) - S_B*B[2])
    energy_for_spin_2 = monte_carlo.compute_energy_for_spin(
        np.array([1, 0, 0]), contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_2, J * S_A * (np.cos(2*np.pi*4/10) + np.cos(2*np.pi*5/10)))

    # to make sure the function runs without runtime errors
    monte_carlo.Metropolis_update(
        spin_config.reshape((2*lattice_sizes[0], 3)), interactions_by_sublattice, 
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J))




def test_monte_carlo_honeycomb_DMI():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()
    np.random.seed(1)

    lattice_sizes = (10, 5)

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    spin_config[:, :, 0, :] = S_A * np.random.rand(*lattice_sizes, 3)
    spin_config[:, :, 1, :] = S_B * np.random.rand(*lattice_sizes, 3)

    interactions_by_sublattice = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.num_sites_unit_cell)
    assert len(interactions_by_sublattice) == model.lattice.num_sites_unit_cell
    assert all(map(lambda x: len(x) == 9, interactions_by_sublattice))
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice[0]))
    # Heisenberg
    bravais_coords_A0, subl_idxs_A0, int_tensor_A0 = interactions_by_sublattice[0][0]
    assert np.allclose(bravais_coords_A0, np.array([[0, 0]]))
    assert np.allclose(subl_idxs_A0, np.array([1]))
    assert np.allclose(int_tensor_A0, J*np.eye(3))
    bravais_coords_A1, subl_idxs_A1, int_tensor_A1 = interactions_by_sublattice[0][1]
    assert np.allclose(bravais_coords_A1, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_A1, np.array([1]))
    assert np.allclose(int_tensor_A1, J*np.eye(3))
    bravais_coords_A2, subl_idxs_A2, int_tensor_A2 = interactions_by_sublattice[0][2]
    assert np.allclose(bravais_coords_A2, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_A2, np.array([1]))
    assert np.allclose(int_tensor_A2, J*np.eye(3))
    # DMI
    bravais_coords_A3, subl_idxs_A3, int_tensor_A3 = interactions_by_sublattice[0][3]
    assert np.allclose(bravais_coords_A3, np.array([[1, -1]]))
    assert np.allclose(subl_idxs_A3, np.array([0]))
    assert np.allclose(int_tensor_A3, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A4, subl_idxs_A4, int_tensor_A4 = interactions_by_sublattice[0][4]
    assert np.allclose(bravais_coords_A4, np.array([[-1, 1]]))
    assert np.allclose(subl_idxs_A4, np.array([0]))
    assert np.allclose(int_tensor_A4, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_A5, subl_idxs_A5, int_tensor_A5 = interactions_by_sublattice[0][5]
    assert np.allclose(bravais_coords_A5, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_A5, np.array([0]))
    assert np.allclose(int_tensor_A5, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A6, subl_idxs_A6, int_tensor_A6 = interactions_by_sublattice[0][6]
    assert np.allclose(bravais_coords_A6, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_A6, np.array([0]))
    assert np.allclose(int_tensor_A6, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_A7, subl_idxs_A7, int_tensor_A7 = interactions_by_sublattice[0][7]
    assert np.allclose(bravais_coords_A7, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_A7, np.array([0]))
    assert np.allclose(int_tensor_A7, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A8, subl_idxs_A8, int_tensor_A8 = interactions_by_sublattice[0][8]
    assert np.allclose(bravais_coords_A8, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_A8, np.array([0]))
    assert np.allclose(int_tensor_A8, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    #
    assert all(map(lambda x: len(x) == 3, interactions_by_sublattice[1]))
    # Heisenberg
    bravais_coords_B0, subl_idxs_B0, int_tensor_B0 = interactions_by_sublattice[1][0]
    assert np.allclose(bravais_coords_B0, np.array([[0, 0]]))
    assert np.allclose(subl_idxs_B0, np.array([0]))
    assert np.allclose(int_tensor_B0, J*np.eye(3))
    bravais_coords_B1, subl_idxs_B1, int_tensor_B1 = interactions_by_sublattice[1][1]
    assert np.allclose(bravais_coords_B1, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_B1, np.array([0]))
    assert np.allclose(int_tensor_B1, J*np.eye(3))
    bravais_coords_B2, subl_idxs_B2, int_tensor_B2 = interactions_by_sublattice[1][2]
    assert np.allclose(bravais_coords_B2, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_B2, np.array([0]))
    assert np.allclose(int_tensor_B2, J*np.eye(3))
    # DMI
    bravais_coords_B3, subl_idxs_B3, int_tensor_B3 = interactions_by_sublattice[1][3]
    assert np.allclose(bravais_coords_B3, np.array([[1, -1]]))
    assert np.allclose(subl_idxs_B3, np.array([1]))
    assert np.allclose(int_tensor_B3, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B4, subl_idxs_B4, int_tensor_B4 = interactions_by_sublattice[1][4]
    assert np.allclose(bravais_coords_B4, np.array([[-1, 1]]))
    assert np.allclose(subl_idxs_B4, np.array([1]))
    assert np.allclose(int_tensor_B4, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_B5, subl_idxs_B5, int_tensor_B5 = interactions_by_sublattice[1][5]
    assert np.allclose(bravais_coords_B5, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_B5, np.array([1]))
    assert np.allclose(int_tensor_B5, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B6, subl_idxs_B6, int_tensor_B6 = interactions_by_sublattice[1][6]
    assert np.allclose(bravais_coords_B6, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_B6, np.array([1]))
    assert np.allclose(int_tensor_B6, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_B7, subl_idxs_B7, int_tensor_B7 = interactions_by_sublattice[1][7]
    assert np.allclose(bravais_coords_B7, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_B7, np.array([1]))
    assert np.allclose(int_tensor_B7, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B8, subl_idxs_B8, int_tensor_B8 = interactions_by_sublattice[1][8]
    assert np.allclose(bravais_coords_B8, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_B8, np.array([1]))
    assert np.allclose(int_tensor_B8, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))

    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4, 2]), 1,
        interactions_by_sublattice, spin_config.reshape((2*np.prod(lattice_sizes), 3)), 
        lattice_sizes, model.lattice.num_sites_unit_cell)
    assert np.allclose(contracted_interactions_for_spin, np.array([
        J*spin_config[4, 2, 0], J*spin_config[4, 1, 0], J*spin_config[3, 2, 0],
        -D*np.array([spin_config[5, 1, 1, 1], -spin_config[5, 1, 1, 0], 0]),
        -D*np.array([-spin_config[3, 3, 1, 1], spin_config[3, 3, 1, 0], 0]),
        -D*np.array([spin_config[3, 2, 1, 1], -spin_config[3, 2, 1, 0], 0]),
        -D*np.array([-spin_config[5, 2, 1, 1], spin_config[5, 2, 1, 0], 0]),
        -D*np.array([spin_config[4, 3, 1, 1], -spin_config[4, 3, 1, 0], 0]),
        -D*np.array([-spin_config[4, 1, 1, 1], spin_config[4, 1, 1, 0], 0]),
    ]))

    energy_for_spin_1 = monte_carlo.compute_energy_for_spin(
        spin_config[4, 2, 1], contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_1, 
        J * spin_config[4, 2, 1].dot(spin_config[4, 2, 0] + spin_config[4, 1, 0] + spin_config[3, 2, 0])) + \
        -D * np.array([0, 0, 1]).dot(np.cross(spin_config[4, 2, 1], spin_config[5, 1, 1] - spin_config[3, 3, 1] + spin_config[3, 2, 1] - spin_config[5, 2, 1] + spin_config[4, 3, 1] - spin_config[4, 1, 1]))

    energy_for_spin_2 = monte_carlo.compute_energy_for_spin(
        np.array([1, 0, 0]), contracted_interactions_for_spin)
    assert np.allclose(energy_for_spin_2, 
        J * (spin_config[4, 2, 0, 0] + spin_config[4, 1, 0, 0] + spin_config[3, 2, 0, 0])) + \
        -D * np.array([0, 0, 1]).dot(np.cross(np.array([1, 0, 0]), spin_config[5, 1, 1] - spin_config[3, 3, 1] + spin_config[3, 2, 1] - spin_config[5, 2, 1] + spin_config[4, 3, 1] - spin_config[4, 1, 1]))

    # to make sure the function runs without runtime errors
    monte_carlo.Metropolis_update(
        spin_config.reshape((2*np.prod(lattice_sizes), 3)), interactions_by_sublattice, 
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J))




def test_util():
    assert util.convert_to_flat_index(np.array([6, 4]), 1, (10, 5,), 3) == 6*5*3 + 4*3 + 1

    np.random.seed(1)
    A = np.random.rand(3, 3, 3, 3)
    b = np.random.rand(2, 3)

    assert np.allclose(util.tensor_contract(A, b), np.einsum("ijkl,k,l", A, *b))
    