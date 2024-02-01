import numpy as np
from magpy.largeS import eigenspace
from magpy.largeS import momentum_space
from magpy.largeS import LSWT
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from magpy.momenta_utils import MSQ, Momenta
from . import test_models


def assert_all_eigenspace_Hamiltonians_equal(ks, eigvs, magnon_Hs_mom_space, expected_magnon_Hs):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        magnon_H = eigenspace.compute_magnon_Hamiltonian(eigvs[order], magnon_Hs_mom_space[order])
        if order == 2:
            assert np.allclose(magnon_H + magnon_H.T.conj(), expected_magnon_H + expected_magnon_H.T.conj())    # symmetrize q, -q
        else:
            assert np.allclose(magnon_H, expected_magnon_H)


def assert_all_commutator_terms_equal(model, ks, eigvs, ks_BZ, eigvs_BZ, eigvs_minus_BZ, expected_commutator_terms):
    for order, expected_commutator_term in enumerate(expected_commutator_terms):
        commutator_term = eigenspace.compute_commutator_term_with_permutations(model, ks[:max(order-1, 0)], eigvs[order], ks_BZ, eigvs_BZ, eigvs_minus_BZ)
        assert np.allclose(commutator_term, expected_commutator_term)

        

# def assert_all_eigenspace_Hamiltonians_equal_for_multiple_ks(model, k_arrays, expected_magnon_Hs):
#     for order, expected_magnon_H in enumerate(expected_magnon_Hs):
#         if order == 0: 
#             continue
#         magnon_H = eigenspace.compute_magnon_Hamiltonians(model, k_arrays[:order-1])
#         assert np.allclose(magnon_H, expected_magnon_H)


def get_eigensystems(model, ks):
    order = len(ks) + 1
    eigws = np.zeros((order, 2*model.lattice.num_sites_unit_cell))
    eigvs = np.zeros((order, *((2*model.lattice.num_sites_unit_cell,) * 2)), dtype=np.complex128)
    eigws[0], eigvs[0] = LSWT.get_eigensystem_momentum_space(model, -np.sum(ks, axis=0))
    for n in range(1, order):
        eigws[n], eigvs[n] = LSWT.get_eigensystem_momentum_space(model, ks[n-1])

    return eigws, eigvs


def test_momentum_space_Hamiltonian_AFM_Heisenberg_chain():
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

    # order S^(1/2)
    expected_magnon_H_3 = np.zeros((4, 4, 4), dtype=np.complex128)
    magnon_H_mom_space_3 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:2])
    eigws_3, eigvs_3 = get_eigensystems(model, ks[:2])

    # order S^0
    magnon_H_mom_space_4 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks)
    eigws_4, eigvs_4 = get_eigensystems(model, ks)
    def expected_magnon_H_4(l, p, q):
        k = -l-p-q
        expected_magnon_H_4 = np.zeros((4, 4, 4, 4), dtype=np.complex128)
        expected_magnon_H_4[1, 0, 3, 2] = -J * (1 + np.exp(-1j*(p+q)[0]))
        expected_magnon_H_4[0, 3, 2, 2] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(l+p+q)[0]))
        expected_magnon_H_4[1, 0, 0, 2] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*q[0]))
        expected_magnon_H_4[3, 3, 2, 1] = J/4 * np.sqrt(S_A/S_B) * (1 + np.exp(-1j*(k+l+p)[0]))
        expected_magnon_H_4[3, 1, 1, 0] = J/4 * np.sqrt(S_B/S_A) * (1 + np.exp(-1j*k[0]))
        return expected_magnon_H_4
 
    assert_all_eigenspace_Hamiltonians_equal(ks, [eigvs_0, eigvs_1, eigvs_2, eigvs_3, ], [
        magnon_H_mom_space_0, 
        magnon_H_mom_space_1,
        magnon_H_mom_space_2,
        magnon_H_mom_space_3,
        #magnon_H_mom_space_4,
    ], [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2(*ks[0:1]),
        expected_magnon_H_3, 
        #expected_magnon_H_4(*ks),
    ])

    


def test_momentum_space_Hamiltonian_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()

    delta_ij = np.array([1])

    np.random.seed(1)
    ks = np.random.rand(3, 1)

    # order S^2
    magnon_H_mom_space_0 = momentum_space.compute_magnon_Hamiltonian(model, ks[:0])
    eigws_0, eigvs_0 = np.zeros((0, 4)), np.zeros((0, 4, 4))
    expected_magnon_H_0 = np.array(J*S**2, dtype=np.complex128)

    # order S^(3/2)
    magnon_H_mom_space_1 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:0])
    eigws_1, eigvs_1 = get_eigensystems(model, np.zeros((0, 1)))
    expected_magnon_H_1 = np.zeros((2,), dtype=np.complex128)

    # order S^1
    magnon_H_mom_space_2 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:1])
    eigws_2, eigvs_2 = get_eigensystems(model, ks[:1])
    def expected_magnon_H_2(q):
        return J*S*np.array([
            [0, np.exp(1j*q.dot(delta_ij)) + np.exp(-1j*q.dot(delta_ij))],
            [-2, 0]
        ], dtype=np.complex128)

    # order S^(1/2)
    magnon_H_mom_space_3 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:2])
    eigws_3, eigvs_3 = get_eigensystems(model, ks[:2])
    expected_magnon_H_3 = np.zeros((2, 2, 2), dtype=np.complex128)

    # order S^0
    magnon_H_mom_space_4 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:3])
    eigws_4, eigvs_4 = get_eigensystems(model, ks)
    def expected_magnon_H_4(k, p, q):
        expected_magnon_H_4 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        expected_magnon_H_4[1, 0, 1, 0] = J*np.exp(1j*(p+q).dot(delta_ij))
        expected_magnon_H_4[1, 0, 0, 1] = -J/4 * (np.exp(1j*q.dot(delta_ij)) + np.exp(1j*(-q).dot(delta_ij)))
        expected_magnon_H_4[0, 1, 1, 0] = -J/4 * (np.exp(1j*(k+p+q).dot(delta_ij)) + np.exp(1j*(-k-p-q).dot(delta_ij)))
        return expected_magnon_H_4

    assert_all_eigenspace_Hamiltonians_equal(ks, [eigvs_0, eigvs_1, eigvs_2, eigvs_3, eigvs_4], [
        magnon_H_mom_space_0, 
        magnon_H_mom_space_1,
        magnon_H_mom_space_2,
        magnon_H_mom_space_3,
        magnon_H_mom_space_4,
    ], [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2(*ks[0:1]),
        expected_magnon_H_3, 
        expected_magnon_H_4(*ks),
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

    assert_all_eigenspace_Hamiltonians_equal(ks, [eigvs_0, eigvs_1,], [
        magnon_H_mom_space_0, 
        magnon_H_mom_space_1,
        #magnon_H_mom_space_2,
        #magnon_H_mom_space_3,
        #magnon_H_mom_space_4,
    ], [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        #expected_magnon_H_2(*ks[0:1]),
        #expected_magnon_H_3, 
        #expected_magnon_H_4(*ks),
    ])

