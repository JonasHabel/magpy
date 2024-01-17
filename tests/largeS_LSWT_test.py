import numpy as np
from magpy.largeS import LSWT
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from . import test_models


def assert_LSWT_BdG_Hamiltonians_equal(model, k, expected_H_LSWT_BdG):
    H_LSWT_BdG = LSWT.compute_LSWT_Hamiltonian_momentum_space_BdG(model, k)
    assert np.allclose(H_LSWT_BdG, expected_H_LSWT_BdG)



def test_momentum_space_Hamiltonian_AFM_Heisenberg_chain():
    model, (J, S_A, S_B) = test_models.FM_Heisenberg_chain()

    np.random.seed(1)
    q = np.random.rand(1)

    expected_H_LSWT_BdG = J*np.array([
        [S_B, 0, 0, -1/2*(1 + np.exp(-1j*q[0]))*np.sqrt(S_A*S_B)],
        [0, S_B, -1/2*(1 + np.exp(-1j*q[0]))*np.sqrt(S_A*S_B), 0],
        [0, -1/2*(1 + np.exp(1j*q[0]))*np.sqrt(S_A*S_B), S_A, 0],
        [-1/2*(1 + np.exp(1j*q[0]))*np.sqrt(S_A*S_B), 0, 0, S_A],
    ])

    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)



def test_momentum_space_Hamiltonian_FM_Heisenberg_chain():
    model, (J, S) = test_models.AFM_Heisenberg_chain()

    delta_ij = np.array([1])

    np.random.seed(1)
    q = np.random.rand(1)

    expected_H_LSWT_BdG = J*S*np.diag([-1 + np.cos(q.dot(delta_ij))] * 2)

    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)




def test_momentum_space_Hamiltonian_honeycomb_DMI():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()

    nns = np.array([
        [0, 0], [-np.sqrt(3)/2, 3/2], [np.sqrt(3)/2, 3/2],
    ])
    nnns = np.array([
        [-np.sqrt(3), 0], [np.sqrt(3)/2, -3/2], [np.sqrt(3)/2, 3/2],
    ])

    np.random.seed(1)
    q = np.random.rand(2)

    # order S^1
    beta_1 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nn)) for nn in nns]))
    beta_2 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nnn)) for nnn in nnns]))
    A = lambda K: J*beta_1(K)*np.sqrt(S_A*S_B)
    B = lambda K: 1j*D*np.cos(theta)*(beta_2(K) - beta_2(-K))
    expected_H_LSWT_BdG = 0.5 * np.array([
        [-3*J*S_B + B(-q)*S_A, 0, A(q), 0],
        [0, -3*J*S_B + B(q)*S_A, 0, A(q)],
        [A(-q), 0, -3*J*S_A - B(-q)*S_B, 0],
        [0, A(-q), 0, -3*J*S_A - B(q)*S_B],
    ])

    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)

    