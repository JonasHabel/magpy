import numpy as np
from magpy.largeS import eigenspace
from magpy.largeS import momentum_space
from magpy.largeS import LSWT
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from magpy.momenta_utils import MSQ, Momenta
from . import test_models


def assert_all_eigenspace_Hamiltonians_equal(ks, eigvs, magnon_Hs_mom_space, expected_magnon_Hs, symmetrize_LSWT=True):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        magnon_H = eigenspace.compute_magnon_Hamiltonian(eigvs[order], magnon_Hs_mom_space[order])
        if order == 2:
            if symmetrize_LSWT:
                assert np.allclose(magnon_H + magnon_H.T.conj(), expected_magnon_H + expected_magnon_H.T.conj())    # symmetrize q, -q
            else:
                assert np.allclose(magnon_H, expected_magnon_H)
        else:
            assert np.allclose(magnon_H, expected_magnon_H)

        

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


def test_eigenspace_Hamiltonian_AFM_Heisenberg_chain():
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

    


def test_eigenspace_Hamiltonian_FM_Heisenberg_chain():
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




def test_eigenspace_Hamiltonian_honeycomb_DMI():
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
    magnon_H_mom_space_2 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:1])
    eigws_2, eigvs_2 = get_eigensystems(model, ks[:1])
    expected_magnon_H_2 = eigvs_2[0].T @ magnon_H_mom_space_2 @ eigvs_2[1]

    # order S^(1/2)
    

    # order S^0

    assert_all_eigenspace_Hamiltonians_equal(ks, [eigvs_0, eigvs_1, eigvs_2, ], [
        magnon_H_mom_space_0, 
        magnon_H_mom_space_1,
        magnon_H_mom_space_2,
        #magnon_H_mom_space_3,
        #magnon_H_mom_space_4,
    ], [
        expected_magnon_H_0, 
        expected_magnon_H_1, 
        expected_magnon_H_2,
        #expected_magnon_H_3, 
        #expected_magnon_H_4(*ks),
    ], symmetrize_LSWT=False)







def test_eigenspace_Hamiltonian_KH_model_2d():
    model, (S, J, K, Gamma, Gamma_prime, J_3, B) = test_models.KH_model_2d()

    # nns = np.array([
    #     [0, 0], [-np.sqrt(3)/2, 3/2], [np.sqrt(3)/2, 3/2],
    # ])
    # nnns = np.array([
    #     [-np.sqrt(3), 0], [np.sqrt(3)/2, -3/2], [np.sqrt(3)/2, 3/2],
    # ])

    np.random.seed(1)
    ks = np.random.rand(4, 2)

    # # order S^2
    # magnon_H_mom_space_0 = momentum_space.compute_magnon_Hamiltonian(model, ks[:0])
    # eigws_0, eigvs_0 = np.zeros((0, 4)), np.zeros((0, 4, 4))
    # expected_magnon_H_0 = np.array(3*J*S_A*S_B, dtype=np.complex128)

    # # order S^(3/2)
    # magnon_H_mom_space_1 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:0])
    # eigws_1, eigvs_1 = get_eigensystems(model, np.zeros((0, 2)))
    # expected_magnon_H_1 = np.zeros((4,))

    # order S^1
    magnon_H_mom_space_2 = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(model, ks[:1])
    eigws_2, eigvs_2 = get_eigensystems(model, ks[:1])
    #eigvs_2[0, :, 0] *= np.exp(1j*np.pi/2)
    eigws_Gamma, eigvs_Gamma = get_eigensystems(model, np.zeros((0, 2)))
    eigvs_Gamma[0, :, 1] *= 1.0j
    sigma_x = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    sigma_y = np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]]))
    sigma_z = np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))
    magnon_H_eigenspace = eigenspace.compute_magnon_Hamiltonian(eigvs_2, magnon_H_mom_space_2)
    
    from magpy.largeS import real_space, normal_order
    cubic_verts_real_space = \
        real_space.compute_magnon_Hamiltonian(model, 3)
    cubic_verts_mom_space = \
        momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
            model, np.array([np.zeros(2), ks[0]]), cubic_verts_real_space)
    cubic_verts_eigenspace = \
        eigenspace.compute_magnon_Hamiltonian_with_permutations(
            np.array([eigvs_2[0], eigvs_Gamma[0], eigvs_2[1]]), cubic_verts_mom_space)
    cubic_verts_eigenspace_nosym = \
        normal_order.normal_order_and_symmetrize_magnon_Hamiltonian(
            cubic_verts_eigenspace)
    
    gauge_phase_k = np.diag(sigma_x @ np.linalg.inv(eigvs_2[1]).conj() @ sigma_x @ eigvs_2[0])[::2]
    gauge_phase_Gamma = np.diag(sigma_x @ np.linalg.inv(eigvs_Gamma[0]).conj() @ sigma_x @ eigvs_Gamma[0])[::2]
    gauge_phase_minusk = np.diag(sigma_x @ np.linalg.inv(eigvs_2[0]).conj() @ sigma_x @ eigvs_2[1])[::2]
    assert np.allclose(
        cubic_verts_eigenspace_nosym[0b000], 
        np.einsum(
            "i,j,k,ijk->ijk",
            gauge_phase_k, gauge_phase_Gamma, gauge_phase_minusk,
            np.transpose(cubic_verts_eigenspace_nosym[0b111].conj(), axes=[2, 1, 0])),  # the hermitian conjugate
        )
    

    momenta_BZ = Momenta.of_BZ(model.lattice, (1, 1), trans=lambda k_BZ: np.random.rand(1, 1, 2))#np.array([[k_BZ[0][0] + 0.5*(model.lattice.reciprocal_lattice.reciprocal_vecs[0] + model.lattice.reciprocal_lattice.reciprocal_vecs[1])]]))
    _, eigvs_BZ = LSWT.get_eigensystems_momentum_space(model, momenta_BZ)
    comm_term = \
        normal_order.compute_commutator_term_with_permutations(
            model, np.zeros((0, 2)), eigvs_Gamma, momenta_BZ.k_arrays[0].reshape((1, 2)), eigvs_BZ.raw_quantity[0].reshape((1, 4, 4)), cubic_verts_real_space)
    
    pass

    # order S^(1/2)
    

    # order S^0

    # assert_all_eigenspace_Hamiltonians_equal(ks, [eigvs_0, eigvs_1, eigvs_2, ], [
    #     magnon_H_mom_space_0, 
    #     magnon_H_mom_space_1,
    #     magnon_H_mom_space_2,
    #     #magnon_H_mom_space_3,
    #     #magnon_H_mom_space_4,
    # ], [
    #     expected_magnon_H_0, 
    #     expected_magnon_H_1, 
    #     expected_magnon_H_2,
    #     #expected_magnon_H_3, 
    #     #expected_magnon_H_4(*ks),
    # ], symmetrize_LSWT=False)