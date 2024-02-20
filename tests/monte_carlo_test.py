import numpy as np
from magpy.classical import monte_carlo, util, nested_list
from magpy.plot import monte_carlo_plot
from . import test_models


def test_monte_carlo_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()
    np.random.seed(1)

    lattice_sizes = np.array([10])

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    #spin_config[..., 2] = np.array([S])[np.newaxis, np.newaxis, np.newaxis]
    spin_config[:, 0, :] = S * np.array([
        [np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, lattice_sizes[0], endpoint=False)
    ])

    interactions_by_sublattice, sep_idxs = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.dim, model.lattice.num_sites_unit_cell)
    assert nested_list.len((interactions_by_sublattice, sep_idxs)) == model.lattice.num_sites_unit_cell
    
    bravais_coords_0 = interactions_by_sublattice[0:1].reshape((1, 1))
    subl_idxs_0 = interactions_by_sublattice[1:2]
    int_tensor_0 = interactions_by_sublattice[2:11].reshape((3, 3))
    assert np.allclose(bravais_coords_0, np.array([[1]]))
    assert np.allclose(subl_idxs_0, np.array([0]))
    assert np.allclose(int_tensor_0, J*np.eye(3))
    bravais_coords_1 = interactions_by_sublattice[11:12].reshape((1, 1))
    subl_idxs_1 = interactions_by_sublattice[12:13]
    int_tensor_1 = interactions_by_sublattice[13:22].reshape((3, 3))
    assert np.allclose(bravais_coords_1, np.array([[-1]]))
    assert np.allclose(subl_idxs_1, np.array([0]))
    assert np.allclose(int_tensor_1, J*np.eye(3))
    
    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4]), nested_list.get((interactions_by_sublattice, sep_idxs), 0), 
        spin_config.reshape((lattice_sizes[0], 3)), 
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
    spin_config_flat = spin_config.reshape((lattice_sizes[0], 3))
    monte_carlo.Metropolis_update(
        spin_config_flat, (interactions_by_sublattice, sep_idxs), 
        np.linalg.norm(spin_config_flat, axis=-1),
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J),
        monte_carlo.sphere_samplers.uniform)
    
    update_infos, final_spin_config = monte_carlo.run_monte_carlo(model, 10, spin_config, 1.0*np.abs(J))
    evolved_spin_configs = monte_carlo.reconstruct_spin_config(update_infos, spin_config, num_steps=None, intermediate_steps=True)
    assert np.allclose(final_spin_config, evolved_spin_configs[-1])

    accepted_update_infos = monte_carlo.get_accepted_updates(update_infos)
    assert len(accepted_update_infos[0].shape) == 2
    assert accepted_update_infos[0].shape[1] == 1
    assert len(accepted_update_infos[1].shape) == 1
    assert len(accepted_update_infos[2].shape) == 2
    assert accepted_update_infos[2].shape[1] == 3



def test_monte_carlo_AFM_Heisenberg_chain():
    B = np.array([0, 0, 1])
    model, (J, S_A, S_B) = test_models.AFM_Heisenberg_chain(B)
    np.random.seed(1)

    lattice_sizes = np.array([10])

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    spin_config[:, 0, :] = S_A * np.array([
        [np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, lattice_sizes[0], endpoint=False)
    ])
    spin_config[:, 1 , 2] = np.array([S_B])[np.newaxis, np.newaxis, np.newaxis]

    interactions_by_sublattice, sep_idxs = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.dim, model.lattice.num_sites_unit_cell)
    assert nested_list.len((interactions_by_sublattice, sep_idxs)) == model.lattice.num_sites_unit_cell
    bravais_coords_A0 = interactions_by_sublattice[0:1].reshape((1, 1))
    subl_idxs_A0 = interactions_by_sublattice[1:2]
    int_tensor_A0 = interactions_by_sublattice[2:11].reshape((3, 3))
    assert np.allclose(bravais_coords_A0, np.array([[0]]))
    assert np.allclose(subl_idxs_A0, np.array([1]))
    assert np.allclose(int_tensor_A0, J*np.eye(3))
    bravais_coords_A1 = interactions_by_sublattice[11:12].reshape((1, 1))
    subl_idxs_A1 = interactions_by_sublattice[12:13]
    int_tensor_A1 = interactions_by_sublattice[13:22].reshape((3, 3))
    assert np.allclose(bravais_coords_A1, np.array([[-1]]))
    assert np.allclose(subl_idxs_A1, np.array([1]))
    assert np.allclose(int_tensor_A1, J*np.eye(3))
    bravais_coords_A2 = interactions_by_sublattice[22:22].reshape((0, 1))
    subl_idxs_A2 = interactions_by_sublattice[22:22]
    int_tensor_A2 = interactions_by_sublattice[22:25].reshape((3,))
    assert np.allclose(bravais_coords_A2, np.array([]))
    assert np.allclose(subl_idxs_A2, np.array([]))
    assert np.allclose(int_tensor_A2, -B)
    #
    bravais_coords_B0 = interactions_by_sublattice[25:26].reshape((1, 1))
    subl_idxs_B0 = interactions_by_sublattice[26:27]
    int_tensor_B0 = interactions_by_sublattice[27:36].reshape((3, 3))
    assert np.allclose(bravais_coords_B0, np.array([[0]]))
    assert np.allclose(subl_idxs_B0, np.array([0]))
    assert np.allclose(int_tensor_B0, J*np.eye(3))
    bravais_coords_B1 = interactions_by_sublattice[36:37].reshape((1, 1))
    subl_idxs_B1 = interactions_by_sublattice[37:38]
    int_tensor_B1 = interactions_by_sublattice[38:47].reshape((3, 3))
    assert np.allclose(bravais_coords_B1, np.array([[1]]))
    assert np.allclose(subl_idxs_B1, np.array([0]))
    assert np.allclose(int_tensor_B1, J*np.eye(3))
    bravais_coords_B2 = interactions_by_sublattice[47:47].reshape((0, 1))
    subl_idxs_B2 = interactions_by_sublattice[47:47]
    int_tensor_B2 = interactions_by_sublattice[47:50].reshape((3,))
    assert np.allclose(bravais_coords_B2, np.array([]))
    assert np.allclose(subl_idxs_B2, np.array([]))
    assert np.allclose(int_tensor_B2, -B)

    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4]), nested_list.get((interactions_by_sublattice, sep_idxs), 1), 
        spin_config.reshape((2*lattice_sizes[0], 3)), 
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
    spin_config_flat = spin_config.reshape((2*lattice_sizes[0], 3))
    monte_carlo.Metropolis_update(
        spin_config_flat, (interactions_by_sublattice, sep_idxs), 
        np.linalg.norm(spin_config_flat, axis=-1),
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J),
        monte_carlo.sphere_samplers.uniform)




def test_monte_carlo_honeycomb_DMI():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()
    np.random.seed(1)

    lattice_sizes = np.array([10, 5])

    spin_config = np.zeros((*lattice_sizes, model.lattice.num_sites_unit_cell, 3), dtype=float)
    spin_config[:, :, 0, :] = S_A * np.random.rand(*lattice_sizes, 3)
    spin_config[:, :, 1, :] = S_B * np.random.rand(*lattice_sizes, 3)

    interactions_by_sublattice, sep_idxs = monte_carlo.group_interactions_by_sublattice(
        model.interactions, model.lattice.dim, model.lattice.num_sites_unit_cell)
    assert nested_list.len((interactions_by_sublattice, sep_idxs)) == model.lattice.num_sites_unit_cell
    # Heisenberg
    bravais_coords_A0 = interactions_by_sublattice[0:2].reshape((1, 2))
    subl_idxs_A0 = interactions_by_sublattice[2:3]
    int_tensor_A0 = interactions_by_sublattice[3:12].reshape((3, 3))
    assert np.allclose(bravais_coords_A0, np.array([[0, 0]]))
    assert np.allclose(subl_idxs_A0, np.array([1]))
    assert np.allclose(int_tensor_A0, J*np.eye(3))
    bravais_coords_A1 = interactions_by_sublattice[12:14].reshape((1, 2))
    subl_idxs_A1 = interactions_by_sublattice[14:15]
    int_tensor_A1 = interactions_by_sublattice[15:24].reshape((3, 3))
    assert np.allclose(bravais_coords_A1, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_A1, np.array([1]))
    assert np.allclose(int_tensor_A1, J*np.eye(3))
    bravais_coords_A2 = interactions_by_sublattice[24:26].reshape((1, 2))
    subl_idxs_A2 = interactions_by_sublattice[26:27]
    int_tensor_A2 = interactions_by_sublattice[27:36].reshape((3, 3))
    assert np.allclose(bravais_coords_A2, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_A2, np.array([1]))
    assert np.allclose(int_tensor_A2, J*np.eye(3))
    # DMI
    bravais_coords_A3 = interactions_by_sublattice[36:38].reshape((1, 2))
    subl_idxs_A3 = interactions_by_sublattice[38:39]
    int_tensor_A3 = interactions_by_sublattice[39:48].reshape((3, 3))
    assert np.allclose(bravais_coords_A3, np.array([[1, -1]]))
    assert np.allclose(subl_idxs_A3, np.array([0]))
    assert np.allclose(int_tensor_A3, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A4 = interactions_by_sublattice[48:50].reshape((1, 2))
    subl_idxs_A4 = interactions_by_sublattice[50:51]
    int_tensor_A4 = interactions_by_sublattice[51:60].reshape((3, 3))
    assert np.allclose(bravais_coords_A4, np.array([[-1, 1]]))
    assert np.allclose(subl_idxs_A4, np.array([0]))
    assert np.allclose(int_tensor_A4, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_A5 = interactions_by_sublattice[60:62].reshape((1, 2))
    subl_idxs_A5 = interactions_by_sublattice[62:63]
    int_tensor_A5 = interactions_by_sublattice[63:72].reshape((3, 3))
    assert np.allclose(bravais_coords_A5, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_A5, np.array([0]))
    assert np.allclose(int_tensor_A5, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A6 = interactions_by_sublattice[72:74].reshape((1, 2))
    subl_idxs_A6 = interactions_by_sublattice[74:75]
    int_tensor_A6 = interactions_by_sublattice[75:84].reshape((3, 3))
    assert np.allclose(bravais_coords_A6, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_A6, np.array([0]))
    assert np.allclose(int_tensor_A6, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_A7 = interactions_by_sublattice[84:86].reshape((1, 2))
    subl_idxs_A7 = interactions_by_sublattice[86:87]
    int_tensor_A7 = interactions_by_sublattice[87:96].reshape((3, 3))
    assert np.allclose(bravais_coords_A7, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_A7, np.array([0]))
    assert np.allclose(int_tensor_A7, D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_A8 = interactions_by_sublattice[96:98].reshape((1, 2))
    subl_idxs_A8 = interactions_by_sublattice[98:99]
    int_tensor_A8 = interactions_by_sublattice[99:108].reshape((3, 3))
    assert np.allclose(bravais_coords_A8, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_A8, np.array([0]))
    assert np.allclose(int_tensor_A8, D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    #
    # Heisenberg
    bravais_coords_B0 = interactions_by_sublattice[108:110].reshape((1, 2))
    subl_idxs_B0 = interactions_by_sublattice[110:111]
    int_tensor_B0 = interactions_by_sublattice[111:120].reshape((3, 3))
    assert np.allclose(bravais_coords_B0, np.array([[0, 0]]))
    assert np.allclose(subl_idxs_B0, np.array([0]))
    assert np.allclose(int_tensor_B0, J*np.eye(3))
    bravais_coords_B1 = interactions_by_sublattice[120:122].reshape((1, 2))
    subl_idxs_B1 = interactions_by_sublattice[122:123]
    int_tensor_B1 = interactions_by_sublattice[123:132].reshape((3, 3))
    assert np.allclose(bravais_coords_B1, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_B1, np.array([0]))
    assert np.allclose(int_tensor_B1, J*np.eye(3))
    bravais_coords_B2 = interactions_by_sublattice[132:134].reshape((1, 2))
    subl_idxs_B2 = interactions_by_sublattice[134:135]
    int_tensor_B2 = interactions_by_sublattice[135:144].reshape((3, 3))
    assert np.allclose(bravais_coords_B2, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_B2, np.array([0]))
    assert np.allclose(int_tensor_B2, J*np.eye(3))
    # DMI
    bravais_coords_B3 = interactions_by_sublattice[144:146].reshape((1, 2))
    subl_idxs_B3 = interactions_by_sublattice[146:147]
    int_tensor_B3 = interactions_by_sublattice[147:156].reshape((3, 3))
    assert np.allclose(bravais_coords_B3, np.array([[1, -1]]))
    assert np.allclose(subl_idxs_B3, np.array([1]))
    assert np.allclose(int_tensor_B3, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B4 = interactions_by_sublattice[156:158].reshape((1, 2))
    subl_idxs_B4 = interactions_by_sublattice[158:159]
    int_tensor_B4 = interactions_by_sublattice[159:168].reshape((3, 3))
    assert np.allclose(bravais_coords_B4, np.array([[-1, 1]]))
    assert np.allclose(subl_idxs_B4, np.array([1]))
    assert np.allclose(int_tensor_B4, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_B5 = interactions_by_sublattice[168:170].reshape((1, 2))
    subl_idxs_B5 = interactions_by_sublattice[170:171]
    int_tensor_B5 = interactions_by_sublattice[171:180].reshape((3, 3))
    assert np.allclose(bravais_coords_B5, np.array([[-1, 0]]))
    assert np.allclose(subl_idxs_B5, np.array([1]))
    assert np.allclose(int_tensor_B5, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B6 = interactions_by_sublattice[180:182].reshape((1, 2))
    subl_idxs_B6 = interactions_by_sublattice[182:183]
    int_tensor_B6 = interactions_by_sublattice[183:192].reshape((3, 3))
    assert np.allclose(bravais_coords_B6, np.array([[1, 0]]))
    assert np.allclose(subl_idxs_B6, np.array([1]))
    assert np.allclose(int_tensor_B6, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
    bravais_coords_B7 = interactions_by_sublattice[192:194].reshape((1, 2))
    subl_idxs_B7 = interactions_by_sublattice[194:195]
    int_tensor_B7 = interactions_by_sublattice[195:204].reshape((3, 3))
    assert np.allclose(bravais_coords_B7, np.array([[0, 1]]))
    assert np.allclose(subl_idxs_B7, np.array([1]))
    assert np.allclose(int_tensor_B7, -D*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
    bravais_coords_B8 = interactions_by_sublattice[204:206].reshape((1, 2))
    subl_idxs_B8 = interactions_by_sublattice[206:207]
    int_tensor_B8 = interactions_by_sublattice[207:216].reshape((3, 3))
    assert np.allclose(bravais_coords_B8, np.array([[0, -1]]))
    assert np.allclose(subl_idxs_B8, np.array([1]))
    assert np.allclose(int_tensor_B8, -D*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))

    contracted_interactions_for_spin = monte_carlo.compute_contracted_interactions_for_spin(
        np.array([4, 2]), nested_list.get((interactions_by_sublattice, sep_idxs), 1), 
        spin_config.reshape((2*np.prod(lattice_sizes), 3)), 
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
    spin_config_flat = spin_config.reshape((2*np.prod(lattice_sizes), 3))
    monte_carlo.Metropolis_update(
        spin_config_flat, (interactions_by_sublattice, sep_idxs), 
        np.linalg.norm(spin_config_flat, axis=-1),
        lattice_sizes, model.lattice.num_sites_unit_cell, 1.0*np.abs(J),
        monte_carlo.sphere_samplers.uniform)
    
    update_infos, final_spin_config = monte_carlo.run_monte_carlo(model, 50, spin_config, 1.0*np.abs(J))
    evolved_spin_configs = monte_carlo.reconstruct_spin_config(update_infos, spin_config, num_steps=16, intermediate_steps=True)
    #monte_carlo_plot.plot_monte_carlo_animation(update_infos, spin_config, model.lattice)




def test_util():
    assert util.convert_to_flat_index(np.array([6, 4]), 1, (10, 5,), 3) == 6*5*3 + 4*3 + 1


    np.random.seed(1)
    A = np.random.rand(3, 3, 3, 3)
    b = np.random.rand(2, 3)
    A_dot_b = util.tensor_contract_jit(A, b).reshape(A.shape[:2])

    assert np.allclose(A_dot_b, np.einsum("ijkl,k,l", A, *b))

    A = np.eye(3)
    b = np.random.rand(1, 3)
    A_dot_b = util.tensor_contract_jit(A, b).reshape(A.shape[:1])

    assert np.allclose(A_dot_b, np.einsum("ij,j", A, *b))



def test_nested_list():
    a, b, c, d, e, f, g, h = np.random.rand(8, 2)
    list_ = [[[a, b], [c, d, e]], [[f], [g, h]]]
    nested_list_depth_2 = nested_list.new(list_, depth=2)
    list_flat_depth_2, separator_idxs_depth_2 = nested_list_depth_2
    assert np.all(list_flat_depth_2 == np.array(
        [a, b, c, d, e, f, g, h]
    ))
    assert np.all(separator_idxs_depth_2 == np.array([
        [0, 0, 2], [0, 1, 5], [1, 0, 6], [1, 1, 8]
    ]))
    assert nested_list.len(nested_list_depth_2) == 2
    sublist_0 = nested_list.get(nested_list_depth_2, 0)
    sublist_flat_0, sublist_sep_idxs_0 = sublist_0
    assert np.allclose(sublist_flat_0, np.array([a, b, c, d, e]))
    assert np.allclose(sublist_sep_idxs_0, np.array([[0, 2], [1, 5]]))
    sublist_1 = nested_list.get(nested_list_depth_2, 1)
    sublist_flat_1, sublist_sep_idxs_1 = sublist_1
    assert np.allclose(sublist_flat_1, np.array([f, g, h]))
    assert np.allclose(sublist_sep_idxs_1, np.array([[0, 1], [1, 3]]))

    nested_list_depth_3 = nested_list.new(list_, depth=3)
    list_flat_depth_3, separator_idxs_depth_3 = nested_list_depth_3
    assert np.all(list_flat_depth_3 == np.array([*a, *b, *c, *d, *e, *f, *g, *h]))
    assert np.all(separator_idxs_depth_3 == np.array([
        [0, 0, 0, 2], [0, 0, 1, 4], 
        [0, 1, 0, 6], [0, 1, 1, 8], [0, 1, 2, 10], 
        [1, 0, 0, 12], 
        [1, 1, 0, 14], [1, 1, 1, 16]
    ]))
    
    rand = np.random.rand(3, 3)
    list_ = [
        [
            [np.array([0, 0]), np.array([0]), np.eye(3).reshape(9)], 
            [np.array([1, 0]), np.array([1]), rand.reshape(9)],
        ], [
            [np.array([-1, 0]), np.array([0]), np.diag([123, 456, 789]).reshape(9)],
        ],
    ]
    nested_list_depth_3 = nested_list.new(list_, depth=3)
    list_flat_depth_3, separator_idxs_depth_3 = nested_list_depth_3
    assert np.all(list_flat_depth_3 == np.array([
        0, 0,  0,  1, 0, 0, 0, 1, 0, 0, 0, 1,   1, 0,  1,  *rand.reshape(9),
        -1, 0,  0,  123, 0, 0, 0, 456, 0, 0, 0, 789,
    ]))
    assert np.all(separator_idxs_depth_3 == np.array([
        [0, 0, 0, 2], [0, 0, 1, 3], [0, 0, 2, 12],
        [0, 1, 0, 14], [0, 1, 1, 15], [0, 1, 2, 24],
        [1, 0, 0, 26], [1, 0, 1, 27], [1, 0, 2, 36],
    ]))

    