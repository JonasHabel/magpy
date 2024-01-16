import numpy as np
from magpy.largeS import momentum_space
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from . import test_models


def assert_all_momentum_space_Hamiltonians_equal(model, ks, expected_magnon_Hs):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        magnon_H = momentum_space.compute_magnon_Hamiltonian(model, ks[:order])
        assert np.allclose(magnon_H, expected_magnon_H)

        

def assert_all_momentum_space_Hamiltonians_equal_for_multiple_ks(model, k_arrays, expected_magnon_Hs):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        if order == 0: 
            continue
        magnon_H = momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation(model, k_arrays[:order-1])
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
    expected_magnon_H_2 = lambda k, q: J*np.array([
        [0, 0, -(1 + np.exp(-1j*q[0])) * np.sqrt(S_A*S_B), 0],
        [2*S_B, 0, 0, 0],
        [0, 0, 0, 0],
        [0, -(1 + np.exp(-1j*k[0])) * np.sqrt(S_A*S_B), (1 + np.exp(-1j*(k+q)[0])) * S_A, 0],
    ], dtype=np.complex128)

    # order S^(1/2)
    expected_magnon_H_3 = np.zeros((4, 4, 4), dtype=np.complex128)

    # order S^0
    def expected_magnon_H_4(k, l, p, q):
        expected_magnon_H_4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)
        expected_magnon_H_4[1, 0, 3, 2] = -J * (1 + np.exp(-1j*(p+q)[0]))
        expected_magnon_H_4[0, 3, 2, 2] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(l+p+q)[0]))
        expected_magnon_H_4[1, 0, 0, 2] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*q[0]))
        expected_magnon_H_4[3, 3, 2, 1] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(k+l+p)[0]))
        expected_magnon_H_4[3, 1, 1, 0] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*k[0]))
        return expected_magnon_H_4
 
    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2(*ks[0:2]),
        expected_magnon_H_3, 
        expected_magnon_H_4(*ks),
    ])

    
    num_ks = 3
    k_arrays = np.random.rand(3, num_ks, 1)
    assert_all_momentum_space_Hamiltonians_equal_for_multiple_ks(
        model, k_arrays,
        [
            None,
            np.zeros((1, 4)),
            np.array([
                [expected_magnon_H_2(-k, k) for k in k_arrays[0]],
                [expected_magnon_H_2(k, -k) for k in k_arrays[0]],
            ]),
            np.zeros((6, num_ks, num_ks, 4, 4, 4)),
            np.array([
               [[[expected_magnon_H_4(-k-p-q, k, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, k, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, p, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, p, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, -k-p-q, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, -k-p-q, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, p, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, p, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, q, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, q, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, -k-p-q, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, -k-p-q, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, k, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, k, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, q, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, q, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, -k-p-q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, -k-p-q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, k, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, k, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, p, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, p, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            ]),
        ]
    )



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
    expected_magnon_H_2 = lambda k, q: J*S*np.array([
        [0, np.exp(1j*q.dot(delta_ij)) + np.exp(1j*k.dot(delta_ij))],
        [-(1 + np.exp(1j*(k+q).dot(delta_ij))), 0]
    ], dtype=np.complex128)

    # order S^(1/2)
    expected_magnon_H_3 = np.zeros((2, 2, 2), dtype=np.complex128)

    # order S^0
    def expected_magnon_H_4(k, l, p, q):
        expected_magnon_H_4 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        expected_magnon_H_4[1, 0, 1, 0] = J*np.exp(1j*(p+q).dot(delta_ij))
        expected_magnon_H_4[1, 0, 0, 1] = -J/4 * (np.exp(1j*q.dot(delta_ij)) + np.exp(1j*(k+l+p).dot(delta_ij)))
        expected_magnon_H_4[0, 1, 1, 0] = -J/4 * (np.exp(1j*(l+p+q).dot(delta_ij)) + np.exp(1j*k.dot(delta_ij)))
        return expected_magnon_H_4

    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2(*ks[0:2]),
        expected_magnon_H_3, 
        expected_magnon_H_4(*ks),
    ])


    num_ks = 3
    k_arrays = np.random.rand(3, num_ks, 1)
    assert_all_momentum_space_Hamiltonians_equal_for_multiple_ks(
        model, k_arrays,
        [
            None,
            np.zeros((1, 2)),
            np.array([
                [expected_magnon_H_2(-k, k) for k in k_arrays[0]],
                [expected_magnon_H_2(k, -k) for k in k_arrays[0]],
            ]),
            np.zeros((6, num_ks, num_ks, 2, 2, 2)),
            np.array([
               [[[expected_magnon_H_4(-k-p-q, k, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, k, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, p, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, p, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(-k-p-q, q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, -k-p-q, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, -k-p-q, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, p, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, p, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, q, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(k, q, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, -k-p-q, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, -k-p-q, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, k, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, k, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, q, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(p, q, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, -k-p-q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, -k-p-q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, k, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, k, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, p, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
               [[[expected_magnon_H_4(q, p, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            ]),
        ]
    )



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
    gamma = lambda k: 3 - np.sum(np.array([np.exp(1j*k.dot(nnn)) for nnn in nnns]))
    expected_magnon_H_1 = lambda k: D*np.sin(theta)*1j/np.sqrt(2) * gamma(k) * np.array([
        S_A**(3/2), -S_A**(3/2), -S_B**(3/2), S_B**(3/2),
    ], dtype=np.complex128)

    # order S^1
    beta_1 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nn)) for nn in nns]))
    beta_2 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nnn)) for nnn in nnns]))
    expected_magnon_H_2 = lambda k, q: np.array([
        [0, 1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_A, 0, J*beta_1(q)*np.sqrt(S_A*S_B)],
        [-3*J*S_B, 0, 0, 0],
        [0, J*beta_1(k)*np.sqrt(S_A*S_B), 0, -1j*D*np.cos(theta)*(beta_2(q) - beta_2(k))*S_B],
        [0, 0, -J*beta_1(k+q)*S_A, 0]
    ], dtype=np.complex128)

    # order S^(1/2)
    prefactor_3 = 1j*D*np.sin(theta) / np.sqrt(2)
    def expected_magnon_H_3(k, p, q):
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
        return expected_magnon_H_3


    # order S^0
    def expected_magnon_H_4(k, l, p, q):
        expected_magnon_H_4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)
        return expected_magnon_H_4

    assert_all_momentum_space_Hamiltonians_equal(model, ks, [
        expected_magnon_H_0, 
        expected_magnon_H_1(*ks[0:1]), 
        expected_magnon_H_2(*ks[0:2]),
        expected_magnon_H_3(*ks[0:3]), 
        # expected_magnon_H_4(*ks),
    ])


    num_ks = 3
    k_arrays = np.random.rand(3, num_ks, 2)
    assert_all_momentum_space_Hamiltonians_equal_for_multiple_ks(
        model, k_arrays[:-1], [
            None,
            np.array([
                expected_magnon_H_1(np.array([0, 0]))
            ]),
            np.array([
                [expected_magnon_H_2(-k, k) for k in k_arrays[0]],
                [expected_magnon_H_2(k, -k) for k in k_arrays[0]],
            ]),
            np.array([
               [[expected_magnon_H_3(-k-q, k, q) for q in k_arrays[1]] for k in k_arrays[0]],
               [[expected_magnon_H_3(-k-q, q, k) for q in k_arrays[1]] for k in k_arrays[0]],
               [[expected_magnon_H_3(k, -k-q, q) for q in k_arrays[1]] for k in k_arrays[0]],
               [[expected_magnon_H_3(k, q, -k-q) for q in k_arrays[1]] for k in k_arrays[0]],
               [[expected_magnon_H_3(q, -k-q, k) for q in k_arrays[1]] for k in k_arrays[0]],
               [[expected_magnon_H_3(q, k, -k-q) for q in k_arrays[1]] for k in k_arrays[0]],
            ]),
            # np.array([
            #    [[[expected_magnon_H_4(-k-p-q, k, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(-k-p-q, k, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(-k-p-q, p, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(-k-p-q, p, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(-k-p-q, q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(-k-p-q, q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, -k-p-q, p, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, -k-p-q, q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, p, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, p, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, q, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(k, q, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, -k-p-q, k, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, -k-p-q, q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, k, -k-p-q, q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, k, q, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, q, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(p, q, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, -k-p-q, k, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, -k-p-q, p, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, k, -k-p-q, p) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, k, p, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, p, -k-p-q, k) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            #    [[[expected_magnon_H_4(q, p, k, -k-p-q) for q in k_arrays[2]] for p in k_arrays[1]] for k in k_arrays[0]],
            # ]),
        ]
    )
