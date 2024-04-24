from magpy import models
from magpy import lattice
from magpy.largeS import LSWT, real_space
from magpy import interactions
import numpy as np

from magpy.momenta_utils import Momenta





def test_quantum_dot():
    latt = lattice.DotLattice(2)
    inter = [
        interactions.HeisenbergInteraction(
            edge=lattice.BravaisLattice.Edge(np.array([]), np.array([0, 1])),
            J=-1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    k_path = lattice.ReciprocalLattice.MomentumPath(np.array([[0]]))
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, np.array([k_path.ks]))
    eigws = eigws[0]

    assert np.allclose(eigws, np.array([0, 0, 2, -2]))


def test_FM_Heisenberg_chain():
    latt = lattice.ChainLattice()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))
    Nk = 100
    k_path = latt.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "Gamma'"], Nk)
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, np.array([k_path.ks]))
    eigws = eigws[0]

    ks = np.linspace(0, 2*np.pi, Nk+1)
    expected_eigws = 2 - 2*np.cos(ks)

    assert np.allclose(eigws[:, 0], expected_eigws)
    assert np.allclose(eigws[:, 1], -expected_eigws)


def test_AFM_Heisenberg_chain():
    latt = lattice.ChainLattice(2)
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, -1]]))
    Nk = 100
    k_path = latt.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "Gamma'"], Nk)
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, np.array([k_path.ks]))
    eigws = eigws[0]

    ks = np.linspace(0, 2*np.pi, Nk+1)
    expected_eigws = np.sqrt(2 - 2*np.cos(ks))
    print(eigws)
    print(expected_eigws)

    assert np.allclose(eigws[:, 0], expected_eigws)
    assert np.allclose(eigws[:, 1], -expected_eigws)
    assert np.allclose(eigws[:, 2], expected_eigws)
    assert np.allclose(eigws[:, 3], -expected_eigws)



def test_Kitaev_interaction():
    J, K = -1.0, -2.0
    latt = lattice.HoneycombLatticeA()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(latt, n=1, J=J),
        interactions.KitaevInteraction(latt, K, ("z", "x", "y")),
    ]
    mod = models.Model(latt, inter, np.array([[1, 1, 1], [1, 1, 1]])/np.sqrt(3))

    gs_rot_mat = mod.compute_ground_state_rotation_matrices()
    assert gs_rot_mat.shape == (2, 3, 3)
    assert np.allclose(gs_rot_mat[0], gs_rot_mat[1])
    assert np.allclose(gs_rot_mat[0][:, 2], np.ones(3) / np.sqrt(3))
    assert np.allclose(gs_rot_mat[0].T @ gs_rot_mat[0], np.identity(3))

    rot_int = mod.compute_rotated_interactions()
    assert len(rot_int) == len(mod.interactions)
    # Heisenberg terms
    for i in range(3):
        assert np.allclose(rot_int[i].interaction_tensor, J*np.array([
            [0, 1/2, 0],
            [1/2, 0, 0],
            [0, 0, 1],
        ]))
    # Kitaev terms
    # z
    assert np.allclose(rot_int[3].interaction_tensor, K*np.array([
        [1/6, 1/6, -1/np.sqrt(18)],
        [1/6, 1/6, -1/np.sqrt(18)],
        [-1/np.sqrt(18), -1/np.sqrt(18), 1/3],
    ]))
    # x
    assert np.allclose(rot_int[4].interaction_tensor, K*np.array([
        [-1/12 + 1j/(2*np.sqrt(12)), 1/6, 1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18))],
        [1/6, -1/12 - 1j/(2*np.sqrt(12)), -1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18))],
        [1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18)), -1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18)), 1/3],
    ]))
    # y
    assert np.allclose(rot_int[5].interaction_tensor, K*np.array([
        [-1/12 - 1j/(2*np.sqrt(12)), 1/6, -1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18))],
        [1/6, -1/12 + 1j/(2*np.sqrt(12)), 1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18))],
        [-1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18)), 1j/(2*np.sqrt(6)) + 1/(2*np.sqrt(18)), 1/3],
    ]))

    H_real_space = real_space.compute_magnon_Hamiltonian(mod, order=2)
    # assert len(H_real_space) == 18
    # # Heisenberg terms ij
    # for i in [0, 3, 6]:
    #     assert np.allclose(H_real_space[i].interaction_tensor, J*np.array([
    #         [0, 1], [1, 0]
    #     ]))
    # # Heisenberg terms ii, jj
    # for i in [1, 2, 4, 5, 7, 8]:
    #     assert np.allclose(H_real_space[i].interaction_tensor, J*np.array([
    #         [0, 0], [-1, 0]
    #     ]))
    # # Kitaev terms
    # # z, ij
    # assert np.allclose(H_real_space[9].interaction_tensor, K*np.array([
    #     [1/3, 1/3],
    #     [1/3, 1/3],
    # ]))
    # # x, ij
    # assert np.allclose(H_real_space[12].interaction_tensor, K*np.array([
    #     [-1/6 + 1j/np.sqrt(12), 1/3],
    #     [1/3, -1/6 - 1j/np.sqrt(12)],
    # ]))
    # # y, ij
    # assert np.allclose(H_real_space[15].interaction_tensor, K*np.array([
    #     [-1/6 - 1j/np.sqrt(12), 1/3],
    #     [1/3, -1/6 + 1j/np.sqrt(12)],
    # ]))
    # # ii, jj
    # for i in [10, 11, 13, 14, 16, 17]:
    #     assert np.allclose(H_real_space[i].interaction_tensor, K*np.array([
    #         [0, 0], [-1/3, 0]
    #     ]))

    my_special_points = {
        "Gamma": np.array([0, 0]),
        "Gamma'": np.array([0, 1]),
        "M": np.array([0, 1/2]),
        "M'": np.array([-1/2, 0]),
        "X": np.array([1/2, -1/2]),
        "Y": np.array([1/2, 1/2]),
    }
    mom_path = latt.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["X", "Gamma", "Y", "Gamma'", "M", "Gamma"], 100, my_special_points
    )
    mom_path = lattice.ReciprocalLattice.MomentumPath(np.array([latt.reciprocal_lattice.reciprocal_vecs[0]]))
    H_mom_space = LSWT.compute_LSWT_Hamiltonians_momentum_space_BdG(mod, np.array([mom_path.ks]), H_real_space)[0]
    a = latt.bravais_vecs
    Delta1 = lambda k: (J + K/3)*(1 + np.exp(1j*np.dot(k, a[0])) + np.exp(1j*np.dot(k, a[1])))
    Delta2 = lambda k: -K/3 + K*(1/6 + 1j/np.sqrt(12))*np.exp(1j*np.dot(k, a[0])) + K*(1/6 - 1j/np.sqrt(12))*np.exp(1j*np.dot(k, a[1]))
    expected_H_mom_space = np.array([[
        [-3*J - K, 0, Delta1(k), Delta2(k)],
        [0, -3*J - K, Delta2(k), Delta1(k)],
        [np.conj(Delta1(k)), np.conj(Delta2(k)), -3*J - K, 0],
        [np.conj(Delta2(k)), np.conj(Delta1(k)), 0, -3*J - K],
    ] for k in mom_path.ks])
    assert np.allclose(H_mom_space, expected_H_mom_space)

    pass


def test_3rd_nn_Heisenberg_interaction():
    J, J_3 = -1.0, 0.5
    latt = lattice.HoneycombLatticeA()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(latt, n=1, J=J),
        interactions.HeisenbergInteraction(lattice.BravaisLattice.Edge(np.array([1, 1]), np.array([0, 1])), J_3),
        interactions.HeisenbergInteraction(lattice.BravaisLattice.Edge(np.array([-1, 1]), np.array([0, 1])), J_3),
        interactions.HeisenbergInteraction(lattice.BravaisLattice.Edge(np.array([1, -1]), np.array([0, 1])), J_3),
    ]
    mod = models.Model(latt, inter, np.array([[1, 1, -2], [1, 1, -2]])/np.sqrt(6))
    
    my_special_points = {
        "Gamma": np.array([0, 0]),
        "Gamma'": np.array([0, 1]),
        "M": np.array([0, 1/2]),
        "M'": np.array([-1/2, 0]),
        "X": np.array([1/2, -1/2]),
        "Y": np.array([1/2, 1/2]),
    }
    mom_path = latt.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["X", "Gamma", "Y", "Gamma'", "M", "Gamma"], 100, my_special_points
    )
    mom_path = lattice.ReciprocalLattice.MomentumPath(np.array([latt.reciprocal_lattice.reciprocal_vecs[0]]))
    H_mom_space = LSWT.compute_LSWT_Hamiltonians_momentum_space_BdG(mod, np.array([mom_path.ks]))[0]
    a = latt.bravais_vecs
    Delta = lambda k: J*(1 + np.exp(1j*np.dot(k, a[0])) + np.exp(1j*np.dot(k, a[1]))) \
                    + J_3*np.sum([np.exp(1j*np.dot(k, a[0] + a[1])) for delta in [a[0] + a[1], a[0] - a[1], -a[0] + a[1]]])
    expected_H_mom_space = np.array([[
        [-3*J - 3*J_3, 0, Delta(k), 0],
        [0, -3*J - 3*J_3, 0, Delta(k)],
        [np.conj(Delta(k)), 0, -3*J - 3*J_3, 0],
        [0, np.conj(Delta(k)), 0, -3*J - 3*J_3],
    ] for k in mom_path.ks])
    assert np.allclose(H_mom_space, expected_H_mom_space)


# TODO finish
def test_stacked_KH_interlayer_interaction():
    latt = lattice.HoneycombLatticeA()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(latt, n=1, J=-1.0)
        # interactions.UniformMagneticField(latt, np.array([0, 0, 2])),
    ]
    classical_gs = np.array([0, 0, 1])
    KH_model_2D = models.Model(latt, inter, np.array([classical_gs]*2))

    J_perp = 1.0
    interlayer_edges = [
        lattice.BravaisLattice.Edge(np.array([0, 0, 0]), np.array([1, 2])),
        lattice.BravaisLattice.Edge(np.array([-1, -1, 0]), np.array([3, 4])),
        lattice.BravaisLattice.Edge(np.array([0, 0, 1]), np.array([5, 0])),
    ]
    stacked_KH_model_3D = models.stack(KH_model_2D, 3, interlayer_edges, [
        interactions.HeisenbergInteraction(edge, J=J_perp) for edge in interlayer_edges
    ], distance_between_layers=1/2, sublattice_shifts=np.array([
        np.array([0, 0, 0]), np.array([0, 0, 0]),   # layer 1
        np.array([0, -1, 0]), np.array([0, -1, 0]),   # layer 2
        np.array([0, 1, 0]), np.array([0, 1, 0]),   # layer 3
    ]))

    mom_path = stacked_KH_model_3D.lattice.reciprocal_lattice.\
        get_momentum_path_approx_equally_spaced(["Gamma", "Z"], 100, {
            "Gamma": np.array([0, 0, 0]),
            "Z": np.array([0, 0, 1]),
        })
    eigw, eigv = LSWT.get_eigensystems_momentum_space(stacked_KH_model_3D, np.array([mom_path.ks]))
    eigw, eigv = eigw[0], eigv[0]
    pass


def test_FM_Heisenberg_stacking():
    latt = lattice.ChainLattice(2)
    latt.sublattices = np.array([[0], [1]])
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]] * 2))
    extended_mod = models.stack(
        mod, 3, [], [], distance_between_layers=2.0, periodic=False
    )

    assert extended_mod.lattice.dim == 1
    assert extended_mod.lattice.embedding_dim == 2
    assert extended_mod.lattice.num_sites_unit_cell == 6
    assert np.allclose(
        extended_mod.lattice.sublattices,
        np.array([
            [0, 0], [1, 0], [0, 2], [1, 2], [0, 4], [1, 4]
        ])
    )
    


def test_FM_Heisenberg_periodic_stacking():
    latt = lattice.ChainLattice(1)
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_1D = models.Model(latt, inter, np.array([[0, 0, 1]]))

    add_edges = [lattice.BravaisLattice.Edge(np.array([0, 1]), np.zeros(2))]
    mod_2D = models.stack(mod_1D, 1, add_edges, [
        interactions.HeisenbergInteraction(add_edges[0], J=-1.0)
    ], periodic=True)

    assert mod_2D.lattice.dim == 2
    assert mod_2D.lattice.embedding_dim == 2

    Nk = 100
    ks = mod_2D.lattice.reciprocal_lattice \
        .sample_inverse_unit_cell([Nk, Nk]) \
        .transpose([1, 2, 0])
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod_2D, Momenta(ks), strip=True)
    eigws, eigvs = eigws.raw_quantity, eigvs.raw_quantity

    a = mod_2D.lattice.bravais_vecs
    expected_eigws = 4 \
        - 2*np.cos(np.tensordot(a[0], ks, axes=[[0], [2]])) \
        - 2*np.cos(np.tensordot(a[1], ks, axes=[[0], [2]]))

    assert np.allclose(eigws[:, :, 0], expected_eigws)
    assert np.allclose(eigws[:, :, 1], -expected_eigws)


def test_FM_Heisenberg_transform():
    test_params = {
        "lattice": [
            lattice.ChainLattice(),
            lattice.ChainLattice(),
            lattice.ChainLattice(2),
            lattice.ChainLattice(2),

            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
            lattice.SquareLattice(),
        ],
        "transf": np.array([
            np.identity(1),
            2*np.identity(1),
            np.identity(1),
            2*np.identity(1),

            np.identity(2),
            np.diag([2, 1]),
            np.diag([1, 2]),
            np.diag([2, 2]), 
            np.diag([3, 2]),
            np.array([[1, 1], [-1, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1], [1, 0]]),
            np.array([[-2, 2], [3, 2]]),
        ]),
        "expected_sublattices": np.array([
            [[0]],
            [[0], [1]],
            [[0], [0.5]],
            [[0], [0.5], [1], [1.5]],

            [[0, 0]],
            [[0, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            [[0, 0], [0, 1]],
            [[0, 0]],
            [[0, 0]],
            [[0, 0], [-1, 1], [0, 1], [1, 1], [-1, 2],
             [0, 2], [1, 2], [2, 2], [0, 3], [1, 3]]
        ]),
        "expected_edges": [
            [
                lattice.BravaisLattice.Edge(np.array([1]), np.array([0, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([1]), np.array([1, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([1]), np.array([1, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([0]), np.array([1, 2])),
                lattice.BravaisLattice.Edge(np.array([0]), np.array([2, 3])),
                lattice.BravaisLattice.Edge(np.array([1]), np.array([3, 0])),
            ],

            [
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([0, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([0, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([1, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([1, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 0])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([1, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([1, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([2, 3])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 2])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([1, 3])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([1, 0])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([3, 2])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([2, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([3, 1])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([1, 2])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([3, 4])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([4, 5])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 3])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([1, 4])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([2, 5])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([2, 0])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([5, 3])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([3, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([4, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([5, 2])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([0, -1]), np.array([0, 1])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([1, 0])),
                lattice.BravaisLattice.Edge(np.array([1, 1]), np.array([1, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([0, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([0, 0])),
                lattice.BravaisLattice.Edge(np.array([-1, 0]), np.array([0, 0])),
            ],
            [
                lattice.BravaisLattice.Edge(np.array([-1, 0]), np.array([0, 4])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 2])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([1, 2])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([1, 4])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([2, 3])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([2, 5])),
                lattice.BravaisLattice.Edge(np.array([-1, 0]), np.array([3, 8])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([3, 6])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([4, 5])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([4, 3])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([5, 6])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([5, 8])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([6, 7])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([6, 9])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([7, 0])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([7, 1])),
                lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([8, 9])),
                lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([8, 7])),
                lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([9, 1])),
                lattice.BravaisLattice.Edge(np.array([1, 1]), np.array([9, 0])),
            ],
        ]
    }

    for x in test_params.values():
        print(len(x))

    for latt, transf, exp_subl, exp_edges in zip(*test_params.values()):
        inter = [
            interactions.NthNearestNeighborHeisenbergInteraction(
                latt, n=1, J=-1.0)
        ]
        mod = models.Model(latt, inter,
            np.tile(np.array([[0, 0, 1]]), (latt.num_sites_unit_cell, 1))
        )
        scaled_mod = models.transform(mod, transf)
        assert np.allclose(scaled_mod.lattice.sublattices, exp_subl)
        assert set(scaled_mod.lattice.edges) == set(exp_edges)



def test_FM_Heisenberg_square_lattice_transform_and_band_structure():
    latt = lattice.SquareLattice()
    mod = models.Model(
        latt,
        [interactions.NthNearestNeighborHeisenbergInteraction(latt, n=1, J=-1)],
        classical_ground_state=np.array([[0, 0, 1]])
    )

    momentum_path = mod.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "X", "M", "Gamma"], 100)
    momenta = Momenta.of_BZ(mod.lattice, (20, 20))
    eigw, _ = LSWT.get_eigensystems_momentum_space(mod, momenta)
    
    from magpy.plot import LSWT_plot
    # LSWT_plot.plot_energies_3D(np.transpose(momenta.k_arrays[0], axes=(2, 0, 1)), eigw.raw_quantity[0])


    trans_1 = np.diag([2, 1])
    mod_1 = models.transform(mod, trans_1)

    momentum_path_1 = mod_1.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "X", "M", "Gamma"], 100, {
            "Gamma": np.array([0, 0]),
            "X": np.array([1, 0]),
            "M": np.array([1, 0.5]),
        })
    eigw_1, _ = LSWT.get_eigensystems_momentum_space(mod_1, Momenta.of(momentum_path_1))
    # LSWT_plot.plot_energies_along_momentum_path(momentum_path_1, eigw_1.raw_quantity[0])
    momentum_path_1a = mod_1.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["X", "Gamma", "Y", "X"], 100, {
            "Gamma": np.array([0, 0]),
            "Y": np.array([0, 0.5]),
            "X": np.array([1, 0]),
        })
    eigw_1a, _ = LSWT.get_eigensystems_momentum_space(mod_1, Momenta.of(momentum_path_1a))
    # LSWT_plot.plot_energies_along_momentum_path(momentum_path_1, eigw_1a.raw_quantity[0])

    assert np.allclose(eigw_1.raw_quantity[0], eigw_1a.raw_quantity[0])
    # momenta_1 = Momenta.of_BZ(mod_1.lattice, (20, 20))
    # eigw_1, _ = LSWT.get_eigensystems_momentum_space(mod_1, momenta_1)
    # LSWT_plot.plot_energies_3D(np.transpose(momenta_1.k_arrays[0], axes=(2, 0, 1)), eigw_1.raw_quantity[0])
    



def test_FM_Heisenberg_square_lattice_delete_dimensions():
    latt = lattice.SquareLattice()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_2D = models.Model(latt, inter, np.array([[0, 0, 1]]))
    mod_1D = models.delete_lattice_dimensions(mod_2D, [1], np.array([[1, 0]]))

    assert mod_1D.lattice.dim == 1
    assert mod_1D.lattice.embedding_dim == 2
    assert mod_1D.lattice.num_sites_unit_cell == 1
    assert set(mod_1D.lattice.edges) == set(
        [lattice.BravaisLattice.Edge(np.array([1]), np.array([0, 0]))]
    )


def test_FM_Heisenberg_chain_lattice_delete_dimensions():
    latt = lattice.ChainLattice(2)
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_1D = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    mod_0D = models.delete_lattice_dimensions(mod_1D, [0], np.zeros((0, 1)))

    assert mod_0D.lattice.num_sites_unit_cell == 2
    assert set(mod_0D.lattice.edges) == set(
        [lattice.BravaisLattice.Edge(np.zeros((0, 1)), np.array([0, 1]))]
    )


def test_FM_Heisenberg_honeycomb_lattice_delete_dimensions():
    latt = lattice.HoneycombLatticeB()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_2D = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    mod_1D = models.delete_lattice_dimensions(mod_2D, [0], np.array([[1, 0]]))

    assert mod_1D.lattice.dim == 1
    assert mod_1D.lattice.embedding_dim == 2
    assert mod_1D.lattice.num_sites_unit_cell == 2
    assert set(mod_1D.lattice.edges) == set([
        lattice.BravaisLattice.Edge(np.array([0]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([1]), np.array([1, 0])),
    ])



def test_FM_Heisenberg_square_lattice_open_bc():
    latt = lattice.SquareLattice()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_2D = models.Model(latt, inter, np.array([[0, 0, 1]]))
    mod_1D = models.add_custom_open_bc(
        mod_2D, [1], np.array([[1, 0]]), np.array([[0, 1]]))

    assert mod_1D.lattice.num_sites_unit_cell == 1
    assert set(mod_1D.lattice.edges) == set([
        lattice.BravaisLattice.Edge(np.array([1]), np.array([0, 0]))
    ])


def test_FM_Heisenberg_chain_lattice_open_bc():
    latt = lattice.ChainLattice(2)
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_1D = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    mod_0D = models.add_custom_open_bc(
        mod_1D, [3], np.zeros((0, 1), dtype=int), np.array([[1]]))

    assert mod_0D.lattice.num_sites_unit_cell == 6
    assert set(mod_0D.lattice.edges) == set([
        lattice.BravaisLattice.Edge(np.array([]), np.array([n, n+1])) \
        for n in range(5)
    ])


def test_FM_Heisenberg_honeycomb_lattice_open_bc_zigzag():
    latt = lattice.HoneycombLatticeB()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod_2D = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    mod_1D = models.add_custom_open_bc(
        mod_2D, [3], np.array([[0, 1]]), np.array([[1, 0]]))

    assert mod_1D.lattice.num_sites_unit_cell == 6
    #assert set(mod_1D.lattice.edges) == set([
    #    lattice.BravaisLattice.Edge(np.array([[0, 0]]), np.array([n, n+1]))
    #])