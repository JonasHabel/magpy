import numpy as np
from magpy import lattice, interactions, models
from magpy.classical import classical
from . import test_models


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



def test_chain_lattice_spin_gradient():
    latt = lattice.ChainLattice(1)

    dimensions = (10,)

    spin_config_FM = np.zeros((*dimensions, 1, 3), dtype=float)
    spin_config_FM[..., 2] = np.array([1])[np.newaxis, np.newaxis]
    spin_gradient_FM = classical.compute_spin_gradient(latt, spin_config_FM)
    expected_spin_gradient_FM = np.zeros((1, *spin_config_FM.shape))
    assert np.allclose(expected_spin_gradient_FM, spin_gradient_FM)

    spin_config_helix = np.array([
        [[np.sin(2*np.pi*x/10), 0, np.cos(2*np.pi*x/10)]] \
        for x in range(dimensions[0])
    ])
    spin_gradient_helix = classical.compute_spin_gradient(latt, spin_config_helix)
    expected_spin_gradient_helix = np.array([[
        [[
            np.sin(2*np.pi*(x+1)/10) - np.sin(2*np.pi*(x-1)/10),
            0,
            np.cos(2*np.pi*(x+1)/10) - np.cos(2*np.pi*(x-1)/10),
        ]] for x in range(dimensions[0])
    ]])
    assert np.allclose(expected_spin_gradient_helix, spin_gradient_helix)



def test_square_lattice_spin_gradient():
    latt = lattice.SquareLattice()

    dimensions = (10, 10)

    spin_config_FM = np.zeros((*dimensions, 1, 3), dtype=float)
    spin_config_FM[..., 2] = np.array([1])[np.newaxis, np.newaxis, np.newaxis]
    spin_gradient_FM = classical.compute_spin_gradient(latt, spin_config_FM)
    expected_spin_gradient_FM = np.zeros((2, *spin_config_FM.shape))
    assert np.allclose(expected_spin_gradient_FM, spin_gradient_FM)

    spin_config_helix = np.array([
        [
            [
                [
                    np.sin(2*np.pi*(x+y)/10), 
                    0, 
                    np.cos(2*np.pi*(x-y)/10)
                ]
            ] for y in range(dimensions[1])
        ] for x in range(dimensions[0])
    ])
    spin_gradient_helix = classical.compute_spin_gradient(latt, spin_config_helix)
    expected_spin_gradient_helix = np.array([
        [   # gradient in x-direction
            [
                [
                    [
                        np.sin(2*np.pi*(x+y+1)/10) - np.sin(2*np.pi*(x+y-1)/10),
                        0,
                        np.cos(2*np.pi*(x-y+1)/10) - np.cos(2*np.pi*(x-y-1)/10),
                    ]
                ] for y in range(dimensions[1])
            ] for x in range(dimensions[0])
        ], [    # gradient in y-direction
            [
                [
                    [
                        np.sin(2*np.pi*(x+y+1)/10) - np.sin(2*np.pi*(x+y-1)/10),
                        0,
                        np.cos(2*np.pi*(x-y-1)/10) - np.cos(2*np.pi*(x-y+1)/10),
                    ]
                ] for y in range(dimensions[1])
            ] for x in range(dimensions[0])
        ],
    ])
    assert np.allclose(expected_spin_gradient_helix, spin_gradient_helix)



def test_square_lattice_skyrmion_density():
    latt = lattice.SquareLattice()

    dimensions = (20, 20)

    spin_config_helix = np.array([
        [
            [
                [
                    np.sin(2*np.pi*(x+y)/20) * np.cos(2*np.pi*(x-y)/20), 
                    np.sin(2*np.pi*(x+y)/20) * np.sin(2*np.pi*(x-y)/20), 
                    np.cos(2*np.pi*(x-y)/20)
                ]
            ] for y in range(dimensions[1])
        ] for x in range(dimensions[0])
    ])
    skyrmion_density_helix = classical.compute_skyrmion_density(latt, spin_config_helix)
    # import matplotlib.pyplot as plt
    # plt.quiver(
    #     *np.meshgrid(np.arange(dimensions[0]), np.arange(dimensions[1])),
    #     spin_config_helix[...,0,0], spin_config_helix[...,0,1], spin_config_helix[...,0,2])
    # plt.show()
    # plt.imshow(skyrmion_density_helix)
    # plt.show()

    quit()
    
    expected_skyrmion_density_helix = np.array([
        [
            [
                np.sin(2*np.pi*(x+y)/20) * np.cos(2*np.pi*(x-y)/20) * () + \
                np.sin(2*np.pi*(x+y)/20) * np.sin(2*np.pi*(x-y)/20) * () + \
                np.cos(2*np.pi*(x-y)/20) * ()
            ] for y in range(dimensions[1])
        ] for x in range(dimensions[0])
    ])
    assert np.allclose(expected_skyrmion_density_helix, skyrmion_density_helix)
