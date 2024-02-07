import numpy as np
from magpy.largeS import normal_order
from magpy.largeS import eigenspace
from magpy.largeS import momentum_space
from magpy.largeS import LSWT
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from magpy.momenta_utils import MSQ, Momenta
from . import test_models




def assert_all_nosym_eigenspace_Hamiltonians_terms_equal(magnon_Hs_eigenspace, expected_magnon_Hs_eigenspace_nosym):
    for order, expected_magnon_H_eigenspace_nosym in enumerate(expected_magnon_Hs_eigenspace_nosym):
        if expected_magnon_H_eigenspace_nosym is None:
            continue
        magnon_H_eigenspace_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonian(magnon_Hs_eigenspace[order])
        assert np.allclose(magnon_H_eigenspace_nosym, expected_magnon_H_eigenspace_nosym)


def assert_all_commutator_terms_equal(model, ks, eigvs, ks_BZ, eigvs_BZ, eigvs_minus_BZ, expected_commutator_terms):
    for order, expected_commutator_term in enumerate(expected_commutator_terms):
        commutator_term = normal_order.compute_commutator_term_with_permutations(model, ks[:max(order-1, 0)], eigvs[order], ks_BZ, eigvs_BZ, eigvs_minus_BZ)
        assert np.allclose(commutator_term, expected_commutator_term)


def get_eigensystems(model, ks):
    order = len(ks) + 1
    eigws = np.zeros((order, 2*model.lattice.num_sites_unit_cell))
    eigvs = np.zeros((order, *((2*model.lattice.num_sites_unit_cell,) * 2)), dtype=np.complex128)
    eigws[0], eigvs[0] = LSWT.get_eigensystem_momentum_space(model, -np.sum(ks, axis=0))
    for n in range(1, order):
        eigws[n], eigvs[n] = LSWT.get_eigensystem_momentum_space(model, ks[n-1])

    return eigws, eigvs


def sum_over_idxs(arr, idxs):
    return sum(arr[idx] for idx in idxs)


def test_normal_order_AFM_Heisenberg_chain():
    model, (J, S_A, S_B) = test_models.AFM_Heisenberg_chain()

    np.random.seed(1)
    ks = np.random.rand(3, 1)

    # order S^2
    magnon_H_mom_space_0 = momentum_space.compute_magnon_Hamiltonian(model, ks[:0])
    eigws_0, eigvs_0 = np.zeros((0, 4)), np.zeros((0, 4, 4))
    expected_magnon_H_0 = np.array(-2*J*S_A*S_B, dtype=np.complex128)

    # order S^(3/2)
    magnon_H_mom_space_1 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:0])
    eigws_1, eigvs_1 = get_eigensystems(model, np.zeros((0, 1)))
    expected_magnon_H_1 = np.zeros((4,), dtype=np.complex128)

    # order S^1
    magnon_H_mom_space_2 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:1])
    eigws_2, eigvs_2 = get_eigensystems(model, ks[:1])
    B = lambda k: -(1 + np.exp(1j*k[0]))*np.sqrt(S_A*S_B)
    tanh = 1/np.abs(B(ks[0])) * (-(S_A + S_B) + np.sqrt((S_A + S_B)**2 - np.abs(B(ks[0]))**2))
    cosh, sinh = 1/np.sqrt(1 - tanh**2), tanh/np.sqrt(1 - tanh**2)
    phase = B(ks[0]) / np.abs(B(ks[0]))
    X = np.array([[cosh, 0, 0, sinh], [0, cosh, sinh, 0], [0, phase*sinh, phase*cosh, 0], [phase*sinh, 0, 0, phase*cosh]])
    def expected_magnon_H_2(q):
        return np.array([
            [0, 2*np.abs(B(q))*sinh*cosh + 2*S_A*sinh**2, np.abs(B(q))*(cosh**2 + sinh**2) + 2*S_A*sinh*cosh, 0],
            [2*S_B*cosh**2, 0, 0, 2*S_B*sinh*cosh],
            [2*S_B*sinh*cosh, 0, 0, 2*S_B*sinh**2],
            [0, np.abs(B(q))*(cosh**2 + sinh**2) + 2*S_A*sinh*cosh, 2*np.abs(B(q))*sinh*cosh + 2*S_A*cosh**2, 0],
        ], dtype=np.complex128)

    


def test_normal_order_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()

    # COMMUTATOR TERMS
    ks_BZ = np.linspace(0, 1, 10).reshape(10, 1)    # not really the BZ, just a mock
    _, eigvs_BZ = LSWT.get_eigensystems_momentum_space(model, Momenta(ks_BZ))
    _, eigvs_minus_BZ = LSWT.get_eigensystems_momentum_space(model, Momenta(-ks_BZ))

    np.random.seed(1)
    ks = np.random.rand(3, 1)

    # order S^2
    eigws_0, eigvs_0 = np.zeros((0, 4)), np.zeros((0, 4, 4))

    # order S^(3/2)
    eigws_1, eigvs_1 = get_eigensystems(model, np.zeros((0, 1)))
    magnon_H_mom_space_1 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(model, ks[:0])
    expected_magnon_H_eigenspace_nosym_1 = np.zeros(2)

    # order S^1
    eigws_2, eigvs_2 = get_eigensystems(model, ks[:1])
    magnon_H_mom_space_2 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(model, ks[:1])
    magnon_H_eigenspace_2 = eigenspace.compute_magnon_Hamiltonian_with_permutations(eigvs_2, magnon_H_mom_space_2)
    expected_magnon_H_eigenspace_nosym_2 = np.array([
        [sum_over_idxs(magnon_H_eigenspace_2, [(0, 0,0), (1, 0,0)])], # a_{-q} a_{q}
        [sum_over_idxs(magnon_H_eigenspace_2, [(0, 0,1), (1, 1,0)])], # a_{-q} a_{-q}^†
        [sum_over_idxs(magnon_H_eigenspace_2, [(0, 1,0), (1, 0,1)])], # a_{q}^† a_{q}
        [sum_over_idxs(magnon_H_eigenspace_2, [(0, 1,1), (1, 1,1)])], # a^†_{q} a^†_{-q}
    ]).reshape((4, 1, 1))
    expected_commutator_terms_0 = np.array([2*J*S*np.sum([np.cos(q[0]) for q in ks_BZ])])

    # order S^(1/2)
    eigws_3, eigvs_3 = get_eigensystems(model, ks[:2])
    magnon_H_mom_space_3 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(model, ks[:2])
    expected_magnon_H_eigenspace_nosym_3 = np.zeros((8, 1, 1, 1))
    expected_commutator_terms_1 = np.zeros((1, 2))

    # order S^0
    eigws_4, eigvs_4 = get_eigensystems(model, ks[:3])
    magnon_H_mom_space_4 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(model, ks[:3])
    magnon_H_eigenspace_4 = eigenspace.compute_magnon_Hamiltonian_with_permutations(eigvs_4, magnon_H_mom_space_4)
    expected_magnon_H_eigenspace_nosym_4 = np.array([
        [1/24 * sum_over_idxs(magnon_H_eigenspace_4, [(n, 0,0,0,0) for n in range(24)])], # a_{-k-p-q}, a_{q} a_{p} a_{k}
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                    # a_{-k-p-q} a_{q} a_{p} a_{k}^†
            (0, 0,0,0,1), (1, 0,0,1,0), (2, 0,0,0,1), (3, 0,0,1,0), (4, 0,1,0,0), (5, 0,1,0,0), 
            (6, 0,0,0,1), (7, 0,0,1,0), (8, 0,0,0,1), (9, 0,0,1,0), (10, 0,1,0,0), (11, 0,1,0,0), 
            (12, 0,0,0,1), (13, 0,0,1,0), (14, 0,0,0,1), (15, 0,0,1,0), (16, 0,1,0,0), (17, 0,1,0,0), 
            (18, 1,0,0,0), (19, 1,0,0,0), (20, 1,0,0,0), (21, 1,0,0,0), (22, 1,0,0,0), (23, 1,0,0,0)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q} a_{p}^† a_{k}
            (0, 0,0,1,0), (1, 0,0,0,1), (2, 0,1,0,0), (3, 0,1,0,0), (4, 0,0,0,1), (5, 0,0,1,0), 
            (6, 0,0,1,0), (7, 0,0,0,1), (8, 0,1,0,0), (9, 0,1,0,0), (10, 0,0,0,1), (11, 0,0,1,0),
            (12, 1,0,0,0), (13, 1,0,0,0), (14, 1,0,0,0), (15, 1,0,0,0), (16, 1,0,0,0), (17, 1,0,0,0), 
            (18, 0,0,0,1), (19, 0,0,1,0), (20, 0,0,0,1), (21, 0,0,1,0), (22, 0,1,0,0), (23, 0,1,0,0)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q} a_{p}^† a_{k}^†
            (0, 0,0,1,1), (1, 0,0,1,1), (2, 0,1,0,1), (3, 0,1,1,0), (4, 0,1,0,1), (5, 0,1,1,0), 
            (6, 0,0,1,1), (7, 0,0,1,1), (8, 0,1,0,1), (9, 0,1,1,0), (10, 0,1,0,1), (11, 0,1,1,0),
            (12, 1,0,0,1), (13, 1,0,1,0), (14, 1,0,0,1), (15, 1,0,1,0), (16, 1,1,0,0), (17, 1,1,0,0), 
            (18, 1,0,0,1), (19, 1,0,1,0), (20, 1,0,0,1), (21, 1,0,1,0), (22, 1,1,0,0), (23, 1,1,0,0)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q}^† a_{p} a_{k}
            (0, 0,1,0,0), (1, 0,1,0,0), (2, 0,0,1,0), (3, 0,0,0,1), (4, 0,0,1,0), (5, 0,0,0,1), 
            (6, 1,0,0,0), (7, 1,0,0,0), (8, 1,0,0,0), (9, 1,0,0,0), (10, 1,0,0,0), (11, 1,0,0,0),
            (12, 0,0,1,0), (13, 0,0,0,1), (14, 0,1,0,0), (15, 0,1,0,0), (16, 0,0,0,1), (17, 0,0,1,0), 
            (18, 0,0,1,0), (19, 0,0,0,1), (20, 0,1,0,0), (21, 0,1,0,0), (22, 0,0,0,1), (23, 0,0,1,0)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q}^† a_{p} a_{k}^†
            (0, 0,1,0,1), (1, 0,1,1,0), (2, 0,0,1,1), (3, 0,0,1,1), (4, 0,1,1,0), (5, 0,1,0,1), 
            (6, 1,0,0,1), (7, 1,0,1,0), (8, 1,0,0,1), (9, 1,0,1,0), (10, 1,1,0,0), (11, 1,1,0,0),
            (12, 0,0,1,1), (13, 0,0,1,1), (14, 0,1,0,1), (15, 0,1,1,0), (16, 0,1,0,1), (17, 0,1,1,0), 
            (18, 1,0,1,0), (19, 1,0,0,1), (20, 1,1,0,0), (21, 1,1,0,0), (22, 1,0,0,1), (23, 1,0,1,0)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q}^† a_{p}^† a_{k}
            (0, 0,1,1,0), (1, 0,1,0,1), (2, 0,1,1,0), (3, 0,1,0,1), (4, 0,0,1,1), (5, 0,0,1,1), 
            (6, 1,0,1,0), (7, 1,0,0,1), (8, 1,1,0,0), (9, 1,1,0,0), (10, 1,0,0,1), (11, 1,0,1,0),
            (12, 1,0,1,0), (13, 1,0,0,1), (14, 1,1,0,0), (15, 1,1,0,0), (16, 1,0,0,1), (17, 1,0,1,0), 
            (18, 0,0,1,1), (19, 0,0,1,1), (20, 0,1,0,1), (21, 0,1,1,0), (22, 0,1,0,1), (23, 0,1,1,0)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q} a_{q}^† a_{p}^† a_{k}^†
            (0, 0,1,1,1), (1, 0,1,1,1), (2, 0,1,1,1), (3, 0,1,1,1), (4, 0,1,1,1), (5, 0,1,1,1), 
            (6, 1,0,1,1), (7, 1,0,1,1), (8, 1,1,0,1), (9, 1,1,1,0), (10, 1,1,0,1), (11, 1,1,1,0),
            (12, 1,0,1,1), (13, 1,0,1,1), (14, 1,1,0,1), (15, 1,1,1,0), (16, 1,1,0,1), (17, 1,1,1,0), 
            (18, 1,0,1,1), (19, 1,0,1,1), (20, 1,1,0,1), (21, 1,1,1,0), (22, 1,1,0,1), (23, 1,1,1,0)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                    # a_{-k-p-q}^† a_{q} a_{p} a_{k}
            (0, 1,0,0,0), (1, 1,0,0,0), (2, 1,0,0,0), (3, 1,0,0,0), (4, 1,0,0,0), (5, 1,0,0,0), 
            (6, 0,1,0,0), (7, 0,1,0,0), (8, 0,0,1,0), (9, 0,0,0,1), (10, 0,0,1,0), (11, 0,0,0,1), 
            (12, 0,1,0,0), (13, 0,1,0,0), (14, 0,0,1,0), (15, 0,0,0,1), (16, 0,0,1,0), (17, 0,0,0,1), 
            (18, 0,1,0,0), (19, 0,1,0,0), (20, 0,0,1,0), (21, 0,0,0,1), (22, 0,0,1,0), (23, 0,0,0,1)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                    # a_{-k-p-q}^† a_{q} a_{p} a_{k}^†
            (0, 1,0,0,1), (1, 1,0,1,0), (2, 1,0,0,1), (3, 1,0,1,0), (4, 1,1,0,0), (5, 1,1,0,0), 
            (6, 0,1,0,1), (7, 0,1,1,0), (8, 0,0,1,1), (9, 0,0,1,1), (10, 0,1,1,0), (11, 0,1,0,1), 
            (12, 0,1,0,1), (13, 0,1,1,0), (14, 0,0,1,1), (15, 0,0,1,1), (16, 0,1,1,0), (17, 0,1,0,1), 
            (18, 1,1,0,0), (19, 1,1,0,0), (20, 1,0,1,0), (21, 1,0,0,1), (22, 1,0,1,0), (23, 1,0,0,1)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q}^† a_{q} a_{p}^† a_{k}
            (0, 1,0,1,0), (1, 1,0,0,1), (2, 1,1,0,0), (3, 1,1,0,0), (4, 1,0,0,1), (5, 1,0,1,0), 
            (6, 0,1,1,0), (7, 0,1,0,1), (8, 0,1,1,0), (9, 0,1,0,1), (10, 0,0,1,1), (11, 0,0,1,1),
            (12, 1,1,0,0), (13, 1,1,0,0), (14, 1,0,1,0), (15, 1,0,0,1), (16, 1,0,1,0), (17, 1,0,0,1), 
            (18, 0,1,0,1), (19, 0,1,1,0), (20, 0,0,1,1), (21, 0,0,1,1), (22, 0,1,1,0), (23, 0,1,0,1)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q}^† a_{q} a_{p}^† a_{k}^†
            (0, 1,0,1,1), (1, 1,0,1,1), (2, 1,1,0,1), (3, 1,1,1,0), (4, 1,1,0,1), (5, 1,1,1,0), 
            (6, 0,1,1,1), (7, 0,1,1,1), (8, 0,1,1,1), (9, 0,1,1,1), (10, 0,1,1,1), (11, 0,1,1,1),
            (12, 1,1,0,1), (13, 1,1,1,0), (14, 1,0,1,1), (15, 1,0,1,1), (16, 1,1,1,0), (17, 1,1,0,1), 
            (18, 1,1,0,1), (19, 1,1,1,0), (20, 1,0,1,1), (21, 1,0,1,1), (22, 1,1,1,0), (23, 1,1,0,1)])
        ],
        [1/4  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q}^† a_{q}^† a_{p} a_{k}
            (0, 1,1,0,0), (1, 1,1,0,0), (2, 1,0,1,0), (3, 1,0,0,1), (4, 1,0,1,0), (5, 1,0,0,1), 
            (6, 1,1,0,0), (7, 1,1,0,0), (8, 1,0,1,0), (9, 1,0,0,1), (10, 1,0,1,0), (11, 1,0,0,1),
            (12, 0,1,1,0), (13, 0,1,0,1), (14, 0,1,1,0), (15, 0,1,0,1), (16, 0,0,1,1), (17, 0,0,1,1), 
            (18, 0,1,1,0), (19, 0,1,0,1), (20, 0,1,1,0), (21, 0,1,0,1), (22, 0,0,1,1), (23, 0,0,1,1)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q}^† a_{q}^† a_{p} a_{k}^†
            (0, 1,1,0,1), (1, 1,1,1,0), (2, 1,0,1,1), (3, 1,0,1,1), (4, 1,1,1,0), (5, 1,1,0,1), 
            (6, 1,1,0,1), (7, 1,1,1,0), (8, 1,0,1,1), (9, 1,0,1,1), (10, 1,1,1,0), (11, 1,1,0,1),
            (12, 0,1,1,1), (13, 0,1,1,1), (14, 0,1,1,1), (15, 0,1,1,1), (16, 0,1,1,1), (17, 0,1,1,1), 
            (18, 1,1,1,0), (19, 1,1,0,1), (20, 1,1,1,0), (21, 1,1,0,1), (22, 1,0,1,1), (23, 1,0,1,1)])
        ],
        [1/6  * sum_over_idxs(magnon_H_eigenspace_4, [                                     # a_{-k-p-q}^† a_{q}^† a_{p}^† a_{k}
            (0, 1,1,1,0), (1, 1,1,0,1), (2, 1,1,1,0), (3, 1,1,0,1), (4, 1,0,1,1), (5, 1,0,1,1), 
            (6, 1,1,1,0), (7, 1,1,0,1), (8, 1,1,1,0), (9, 1,1,0,1), (10, 1,0,1,1), (11, 1,0,1,1),
            (12, 1,1,1,0), (13, 1,1,0,1), (14, 1,1,1,0), (15, 1,1,0,1), (16, 1,0,1,1), (17, 1,0,1,1), 
            (18, 0,1,1,1), (19, 0,1,1,1), (20, 0,1,1,1), (21, 0,1,1,1), (22, 0,1,1,1), (23, 0,1,1,1)])
        ],
        [1/24 * sum_over_idxs(magnon_H_eigenspace_4, [(n, 1,1,1,1) for n in range(24)])], # a_{-k-p-q}^†, a_{q}^† a_{p}^† a_{k}^†
    ]).reshape((16, 1, 1, 1, 1))
    expected_commutator_terms_2 = J * np.array([
        [[0, 0], [np.sum(np.exp(1j*(ks[0,0] + ks_BZ[:,0])) - 2*np.cos(ks_BZ[:,0])), 0]],   # α_{-q} α_{q}
        [[0, 0], [np.sum(np.exp(1j*(-ks[0,0] + ks_BZ[:,0])) - 2*np.cos(ks_BZ[:,0])), 0]],   # α_{q} α_{-q}
    ])

    assert_all_nosym_eigenspace_Hamiltonians_terms_equal([
        None,
        magnon_H_mom_space_1,
        magnon_H_mom_space_2,
        magnon_H_mom_space_3,
        magnon_H_mom_space_4,
    ], [
        None,
        expected_magnon_H_eigenspace_nosym_1,
        expected_magnon_H_eigenspace_nosym_2,
        expected_magnon_H_eigenspace_nosym_3,
        expected_magnon_H_eigenspace_nosym_4,
    ])

    assert_all_commutator_terms_equal(
        model,
        ks, [eigvs_0, eigvs_1, eigvs_2, ],
        ks_BZ, eigvs_BZ.raw_quantity, eigvs_minus_BZ.raw_quantity,
        [
            expected_commutator_terms_0,
            expected_commutator_terms_1,
            expected_commutator_terms_2,
        ])




def test_normal_order_honeycomb_DMI():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()

    nns = np.array([
        [0, 0], [-np.sqrt(3)/2, 3/2], [np.sqrt(3)/2, 3/2],
    ])
    nnns = np.array([
        [-np.sqrt(3), 0], [np.sqrt(3)/2, -3/2], [np.sqrt(3)/2, 3/2],
    ])

    np.random.seed(1)
    ks = np.random.rand(4, 2)

    # order S^2
    magnon_H_mom_space_0 = momentum_space.compute_magnon_Hamiltonian(model, ks[:0])
    eigws_0, eigvs_0 = np.zeros((0, 4)), np.zeros((0, 4, 4))
    expected_magnon_H_0 = np.array(3*J*S_A*S_B, dtype=np.complex128)

    # order S^(3/2)
    magnon_H_mom_space_1 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:0])
    eigws_1, eigvs_1 = get_eigensystems(model, np.zeros((0, 2)))
    expected_magnon_H_1 = np.zeros((4,))

    # order S^1
    beta_1 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nn)) for nn in nns]))
    beta_2 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nnn)) for nnn in nnns]))
    expected_magnon_H_2 = lambda k, q: np.array([
        [0, 1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_A, 0, J*beta_1(q)*np.sqrt(S_A*S_B)],
        [-3*J*S_B, 0, 0, 0],
        [0, J*beta_1(k)*np.sqrt(S_A*S_B), 0, -1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_B],
        [0, 0, -J*beta_1(k+q)*S_A, 0]
    ], dtype=np.complex128)



def test_normal_order_and_symmetrize_one_band_cubic_vertex():
    vertex = np.arange(48).reshape((6, 2, 2, 2)) + 1
    vertex[0, 1, 0, 0] += 990
    vertex[2, 0, 1, 1] += 8800
    expected_nosym_vertex = np.array([
        (1+9+17+25+33+41)/6,    # a_{-k-q}  a_{q}    a_{k}
        (2+11+18+27+37+45)/2,   # a_{-k-q}  a_{q}    a^†_{-k}
        (3+10+21+29+34+43)/2,   # a_{-k-q}  a^†_{-q} a_{k}
        (4+12+22+31+38+47)/2,   # a_{-k-q}  a^†_{-q} a^†_{-k}
        (995+13+19+26+35+42)/2, # a^†_{k+q} a_{q}    a_{k}
        (6+15+8820+28+39+46)/2, # a^†_{k+q} a_{q}    a^†_{-k}
        (7+14+23+30+36+44)/2,   # a^†_{k+q} a^†_{-q} a_{k}
        (8+16+24+32+40+48)/6,   # a^†_{k+q} a^†_{-q} a^†_{-k}
    ]).reshape((8, 1, 1, 1))

    nosym_vertex = normal_order.normal_order_and_symmetrize_magnon_Hamiltonian(vertex)
    assert np.allclose(expected_nosym_vertex, nosym_vertex)
    
    vertices = np.zeros((6, 4, 3, 2, 2, 2))
    vertices[:] = vertex[:, np.newaxis, np.newaxis]
    expected_nosym_vertices = np.zeros((8, 4, 3, 1, 1, 1))
    expected_nosym_vertices[:] = np.array([
        (1+9+17+25+33+41)/6,    # a_{-k-q}  a_{q}    a_{k}
        (2+11+18+27+37+45)/2,   # a_{-k-q}  a_{q}    a^†_{-k}
        (3+10+21+29+34+43)/2,   # a_{-k-q}  a^†_{-q} a_{k}
        (4+12+22+31+38+47)/2,   # a_{-k-q}  a^†_{-q} a^†_{-k}
        (995+13+19+26+35+42)/2, # a^†_{k+q} a_{q}    a_{k}
        (6+15+8820+28+39+46)/2, # a^†_{k+q} a_{q}    a^†_{-k}
        (7+14+23+30+36+44)/2,   # a^†_{k+q} a^†_{-q} a_{k}
        (8+16+24+32+40+48)/6,   # a^†_{k+q} a^†_{-q} a^†_{-k}
    ]).reshape((8, 1, 1, 1))[:, np.newaxis, np.newaxis]

    nosym_vertices = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
        MSQ(vertices, Momenta(np.zeros((4, 2)), np.zeros((3, 2)))))
    assert np.allclose(expected_nosym_vertices, nosym_vertices.raw_quantity)



def test_commutator_terms():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()
    ks = Momenta()     # order 3
    eigvs = MSQ([np.random.rand(1, 4, 4)], Momenta(np.zeros((1, 2))))
    ks_BZ = Momenta.of_BZ(model.lattice, (10, 10))
    eigvs_BZ = np.random.rand(10, 10, 4, 4)
    eigvs_minus_BZ = np.random.rand(10, 10, 4, 4)

    comm_terms = normal_order.compute_commutator_terms_with_permutations(
        model, ks, eigvs, ks_BZ, MSQ(eigvs_BZ, ks_BZ), MSQ(eigvs_minus_BZ, ks_BZ))
    assert comm_terms.raw_quantity.shape == (1, 1, 4)   # 1 permutation, 1 momentum, 4 BdG bands

    comm_terms_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(comm_terms)
    assert comm_terms_nosym.raw_quantity.shape == (2, 1, 2) # 2 permutations, 1 momentum, 2 particle bands