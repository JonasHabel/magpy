import numpy as np
from magpy.largeS import momentum_space
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from . import test_models


def assert_all_momentum_space_Hamiltonians_equal(model, ks, expected_magnon_Hs):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        magnon_H = momentum_space.compute_magnon_Hamiltonian(model, order, ks[:order])
        assert np.allclose(magnon_H, expected_magnon_H)


def test_momentum_space_Hamiltonian_AFM_Heisenberg_chain():
    model, (J, S_A, S_B) = test_models.FM_Heisenberg_chain()

    np.random.seed(1)
    ks = np.random.rand(4, 1)

    # order S^2
    expected_magnon_H_0 = np.array(-2*J*S_A*S_B, dtype=np.complex128)

    # order S^(3/2)
    expected_magnon_H_1 = np.zeros((4,), dtype=np.complex128)

    # order S^1
    k, q = ks[0:2]
    expected_magnon_H_2 = J*np.array([
        [0, 0, -(1 + np.exp(-1j*q[0])) * np.sqrt(S_A*S_B), 0],
        [2*S_B, 0, 0, 0],
        [0, 0, 0, 0],
        [0, -(1 + np.exp(-1j*k[0])) * np.sqrt(S_A*S_B), (1 + np.exp(-1j*(k+q)[0])) * S_A, 0],
    ], dtype=np.complex128)

    # order S^(1/2)
    expected_magnon_H_3 = np.zeros((4, 4, 4), dtype=np.complex128)

    # order S^0
    k, l, p, q = ks
    expected_magnon_H_4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    expected_magnon_H_4[1, 0, 3, 2] = -J * (1 + np.exp(-1j*(p+q)[0]))
    expected_magnon_H_4[0, 3, 2, 2] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(l+p+q)[0]))
    expected_magnon_H_4[1, 0, 0, 2] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*q[0]))
    expected_magnon_H_4[3, 3, 2, 1] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(k+l+p)[0]))
    expected_magnon_H_4[3, 1, 1, 0] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*k[0]))
 
    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2,
        expected_magnon_H_3, 
        expected_magnon_H_4,
    ])


def test_momentum_space_Hamiltonian_FM_Heisenberg_chain():
    model, (J, S) = test_models.AFM_Heisenberg_chain()

    delta_ij = np.array([1])

    np.random.seed(1)
    ks = np.random.rand(4, 1)

    # order S^2
    expected_magnon_H_0 = np.array(J*S**2, dtype=np.complex128)

    # order S^(3/2)
    expected_magnon_H_1 = np.zeros((2,), dtype=np.complex128)

    # order S^1
    k, q = ks[0:2]
    expected_magnon_H_2 = J*S*np.array([
        [0, np.exp(1j*q.dot(delta_ij)) + np.exp(1j*k.dot(delta_ij))],
        [-(1 + np.exp(1j*(k+q).dot(delta_ij))), 0]
    ], dtype=np.complex128)

    # order S^(1/2)
    expected_magnon_H_3 = np.zeros((2, 2, 2), dtype=np.complex128)

    # order S^0
    k, l, p, q = ks
    expected_magnon_H_4 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    expected_magnon_H_4[1, 0, 1, 0] = J*np.exp(1j*(p+q).dot(delta_ij))
    expected_magnon_H_4[1, 0, 0, 1] = -J/4 * (np.exp(1j*q.dot(delta_ij)) + np.exp(1j*(k+l+p).dot(delta_ij)))
    expected_magnon_H_4[0, 1, 1, 0] = -J/4 * (np.exp(1j*(l+p+q).dot(delta_ij)) + np.exp(1j*k.dot(delta_ij)))

    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2,
        expected_magnon_H_3, 
        expected_magnon_H_4,
    ])


def test_momentum_space_Hamiltonian_honeycomb_DMI():
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
    expected_magnon_H_0 = np.array(3*J*S_A*S_B, dtype=np.complex128)

    # order S^(3/2)
    k = ks[0]
    gamma_k = 3 - np.sum(np.array([np.exp(1j*k.dot(nnn)) for nnn in nnns]))
    expected_magnon_H_1 = D*np.sin(theta)*1j/np.sqrt(2) * gamma_k * np.array([
        S_A**(3/2), -S_A**(3/2), -S_B**(3/2), S_B**(3/2),
    ], dtype=np.complex128)

    # order S^1
    k, q = ks[0:2]
    beta_1 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nn)) for nn in nns]))
    beta_2 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nnn)) for nnn in nnns]))
    expected_magnon_H_2 = np.array([
        [0, 1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_A, 0, J*beta_1(q)*np.sqrt(S_A*S_B)],
        [-3*J*S_B, 0, 0, 0],
        [0, J*beta_1(k)*np.sqrt(S_A*S_B), 0, -1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_B],
        [0, 0, -J*beta_1(k+q)*S_A, 0]
    ], dtype=np.complex128)

    # order S^(1/2)
    k, p, q = ks[0:3]
    prefactor_3 = 1j*D*np.sin(theta) / np.sqrt(2)
    expected_magnon_H_3 = np.zeros((4, 4, 4), dtype=np.complex128)
    expected_magnon_H_3[1, 0, 0] = np.sqrt(S_A) * (beta_2(q) + 1/4*(beta_2(k+p+q) - 3))
    expected_magnon_H_3[1, 1, 0] = -np.sqrt(S_A) * (beta_2(k) + 1/4*(beta_2(k+p+q) - 3))
    expected_magnon_H_3[0, 1, 0] = -np.sqrt(S_A) * beta_2(p+q)
    expected_magnon_H_3[1, 0, 1] = np.sqrt(S_A) * beta_2(k+p)
    expected_magnon_H_3[3, 2, 2] = -np.sqrt(S_B) * (beta_2(q) + 1/4*(beta_2(k+p+q) - 3))
    expected_magnon_H_3[3, 3, 2] = np.sqrt(S_B) * (beta_2(k) + 1/4*(beta_2(k+p+q) - 3))
    expected_magnon_H_3[2, 3, 2] = np.sqrt(S_B) * beta_2(p+q)
    expected_magnon_H_3[3, 2, 3] = -np.sqrt(S_B) * beta_2(k+p)
    expected_magnon_H_3 *= prefactor_3


    # order S^0
    k, l, p, q = ks
    expected_magnon_H_4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)

    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2,
        expected_magnon_H_3, 
        # expected_magnon_H_4,
    ])
