import numpy as np
from magpy.largeS import real_space
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from . import test_models


def assert_real_space_Hamiltonian_equal(magnon_H, expected_magnon_H):
    assert len(magnon_H) == len(expected_magnon_H)
    for inter in magnon_H:
        assert inter in expected_magnon_H


def assert_all_real_space_Hamiltonians_equal(model, expected_magnon_Hs):
    for order, expected_magnon_H in enumerate(expected_magnon_Hs):
        magnon_H = real_space.compute_magnon_Hamiltonian(model, order=order)
        assert_real_space_Hamiltonian_equal(magnon_H, expected_magnon_H)    


def test_real_space_Hamiltonian_AFM_Heisenberg_chain():
    model, (J, S_A, S_B) = test_models.AFM_Heisenberg_chain()

    site_k = BravaisLattice.Site(np.array([-1]), 1)
    site_i = BravaisLattice.Site(np.array([0]), 0)
    site_j = BravaisLattice.Site(np.array([0]), 1)

    # order S^2
    expected_magnon_H_0 = [
        Interaction([], np.array(-J*S_A*S_B)),  # ij-bond
        Interaction([], np.array(-J*S_A*S_B)),  # ik-bond
    ]

    # order S^1
    expected_magnon_H_2 = [
        Interaction([site_i]*2, np.array([[0, 0], [S_B*J, 0]])),
        Interaction([site_i]*2, np.array([[0, 0], [S_B*J, 0]])),
        Interaction([site_j]*2, np.array([[0, 0], [S_A*J, 0]])),
        Interaction([site_k]*2, np.array([[0, 0], [S_A*J, 0]])),
        Interaction([site_i, site_j], np.array([[-np.sqrt(S_A*S_B)*J, 0], [0, 0]])),
        Interaction([site_j, site_i], np.array([[0, 0], [0, -np.sqrt(S_A*S_B)*J]])),
        Interaction([site_i, site_k], np.array([[-np.sqrt(S_A*S_B)*J, 0], [0, 0]])),
        Interaction([site_k, site_i], np.array([[0, 0], [0, -np.sqrt(S_A*S_B)*J]])),
    ]

    # order S^0
    expected_int_tensor_iijj = np.zeros((2, 2, 2, 2))
    expected_int_tensor_iijj[1, 0, 1, 0] = -J
    expected_int_tensor_iiij = np.zeros((2, 2, 2, 2))
    expected_int_tensor_iiij[1, 0, 0, 0] = J/4 * np.sqrt(S_B/S_A)
    expected_int_tensor_ijjj = np.zeros((2, 2, 2, 2))
    expected_int_tensor_ijjj[0, 1, 0, 0] = J/4 * np.sqrt(S_A/S_B)
    expected_int_tensor_jiii = np.zeros((2, 2, 2, 2))
    expected_int_tensor_jiii[1, 1, 1, 0] = J/4 * np.sqrt(S_B/S_A)
    expected_int_tensor_jjji = np.zeros((2, 2, 2, 2))
    expected_int_tensor_jjji[1, 1, 0, 1] = J/4 * np.sqrt(S_A/S_B)
    expected_magnon_H_4 = [
        Interaction([site_i]*2 + [site_j]*2, expected_int_tensor_iijj),
        Interaction([site_i]*3 + [site_j], expected_int_tensor_iiij),
        Interaction([site_i] + [site_j]*3, expected_int_tensor_ijjj),
        Interaction([site_j] + [site_i]*3, expected_int_tensor_jiii),
        Interaction([site_j]*3 + [site_i], expected_int_tensor_jjji),
        Interaction([site_i]*2 + [site_k]*2, expected_int_tensor_iijj),
        Interaction([site_i]*3 + [site_k], expected_int_tensor_iiij),
        Interaction([site_i] + [site_k]*3, expected_int_tensor_ijjj),
        Interaction([site_k] + [site_i]*3, expected_int_tensor_jiii),
        Interaction([site_k]*3 + [site_i], expected_int_tensor_jjji),
    ]

    assert_all_real_space_Hamiltonians_equal(model, [
        expected_magnon_H_0, [], expected_magnon_H_2, [], expected_magnon_H_4,
    ])


def test_real_space_Hamiltonian_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()

    site_i = BravaisLattice.Site(np.array([0]), 0)
    site_j = BravaisLattice.Site(np.array([1]), 0)

    # order S^2
    expected_magnon_H_0 = [
        Interaction([], np.array(J*S**2))
    ]

    # order S^1
    expected_magnon_H_2 = [
        Interaction([site_i]*2, np.array([[0, 0], [-S*J, 0]])),
        Interaction([site_j]*2, np.array([[0, 0], [-S*J, 0]])),
        Interaction([site_i, site_j], np.array([[0, S*J], [0, 0]])),
        Interaction([site_j, site_i], np.array([[0, S*J], [0, 0]])),
    ]

    # order S^0
    expected_int_tensor_iijj = np.zeros((2, 2, 2, 2))
    expected_int_tensor_iijj[1, 0, 1, 0] = J
    expected_int_tensor_iiij = np.zeros((2, 2, 2, 2))
    expected_int_tensor_iiij[1, 0, 0, 1] = -J/4
    expected_int_tensor_ijjj = np.zeros((2, 2, 2, 2))
    expected_int_tensor_ijjj[0, 1, 1, 0] = -J/4
    expected_int_tensor_jiii = np.zeros((2, 2, 2, 2))
    expected_int_tensor_jiii[0, 1, 1, 0] = -J/4
    expected_int_tensor_jjji = np.zeros((2, 2, 2, 2))
    expected_int_tensor_jjji[1, 0, 0, 1] = -J/4
    expected_magnon_H_4 = [
        Interaction([site_i]*2 + [site_j]*2, expected_int_tensor_iijj),
        Interaction([site_i]*3 + [site_j], expected_int_tensor_iiij),
        Interaction([site_i] + [site_j]*3, expected_int_tensor_ijjj),
        Interaction([site_j] + [site_i]*3, expected_int_tensor_jiii),
        Interaction([site_j]*3 + [site_i], expected_int_tensor_jjji),
    ]

    assert_all_real_space_Hamiltonians_equal(model, [
        expected_magnon_H_0, [], expected_magnon_H_2, [], expected_magnon_H_4,
    ])


def test_real_space_Hamiltonian_honeycomb_DMI():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb()

    site_iA = BravaisLattice.Site(np.array([0, 0]), 0)
    site_iA1 = BravaisLattice.Site(np.array([-1, 0]), 0)
    site_iA2 = BravaisLattice.Site(np.array([0, -1]), 0)
    site_iA3 = BravaisLattice.Site(np.array([1, -1]), 0)
    site_iA4 = BravaisLattice.Site(np.array([0, 1]), 0)
    site_iB = BravaisLattice.Site(np.array([0, 0]), 1)
    site_iB1 = BravaisLattice.Site(np.array([1, 0]), 1)
    site_iB2 = BravaisLattice.Site(np.array([0, 1]), 1)
    site_iB3 = BravaisLattice.Site(np.array([1, -1]), 1)
    site_iB4 = BravaisLattice.Site(np.array([-1, 0]), 1)

    # order S^2
    expected_magnon_H_0 = [
        # Heisenberg
        Interaction([], np.array(J*S_A*S_B)),
        Interaction([], np.array(J*S_A*S_B)),
        Interaction([], np.array(J*S_A*S_B)),
    ]

    # order S^(3/2)
    prefactor_1 = D*np.sin(theta)*1j/np.sqrt(2)
    expected_magnon_H_1 = [
        # DMI-x
        Interaction([site_iA], prefactor_1*S_A**(3/2)*np.array([1, -1])),
        Interaction([site_iA3], -prefactor_1*S_A**(3/2)*np.array([1, -1])),
        Interaction([site_iA], prefactor_1*S_A**(3/2)*np.array([1, -1])),
        Interaction([site_iA1], -prefactor_1*S_A**(3/2)*np.array([1, -1])),
        Interaction([site_iA], prefactor_1*S_A**(3/2)*np.array([1, -1])),
        Interaction([site_iA4], -prefactor_1*S_A**(3/2)*np.array([1, -1])),

        Interaction([site_iB], -prefactor_1*S_B**(3/2)*np.array([1, -1])),
        Interaction([site_iB3], prefactor_1*S_B**(3/2)*np.array([1, -1])),
        Interaction([site_iB], -prefactor_1*S_B**(3/2)*np.array([1, -1])),
        Interaction([site_iB4], prefactor_1*S_B**(3/2)*np.array([1, -1])),
        Interaction([site_iB], -prefactor_1*S_B**(3/2)*np.array([1, -1])),
        Interaction([site_iB2], prefactor_1*S_B**(3/2)*np.array([1, -1])),
    ]

    # order S^1
    prefactor_2 = D*np.cos(theta)*1j
    expected_magnon_H_2 = [
        # Heisenberg -- on-site terms
        Interaction([site_iA]*2, np.array([[0, 0], [-S_B*J, 0]])),
        Interaction([site_iB]*2, np.array([[0, 0], [-S_A*J, 0]])),
        Interaction([site_iA]*2, np.array([[0, 0], [-S_B*J, 0]])),
        Interaction([site_iB1]*2, np.array([[0, 0], [-S_A*J, 0]])),
        Interaction([site_iA]*2, np.array([[0, 0], [-S_B*J, 0]])),
        Interaction([site_iB2]*2, np.array([[0, 0], [-S_A*J, 0]])),

        # Heisenberg -- hopping terms
        Interaction([site_iA, site_iB], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),
        Interaction([site_iB, site_iA], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),
        Interaction([site_iA, site_iB1], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),
        Interaction([site_iB1, site_iA], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),
        Interaction([site_iA, site_iB2], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),
        Interaction([site_iB2, site_iA], np.array([[0, np.sqrt(S_A*S_B)*J], [0, 0]])),

        # DMI-z -- hopping terms
        Interaction([site_iA, site_iA3], prefactor_2*S_A*np.array([[0, 1], [0, 0]])),
        Interaction([site_iA3, site_iA], -prefactor_2*S_A*np.array([[0, 1], [0, 0]])),
        Interaction([site_iA, site_iA1], prefactor_2*S_A*np.array([[0, 1], [0, 0]])),
        Interaction([site_iA1, site_iA], -prefactor_2*S_A*np.array([[0, 1], [0, 0]])),
        Interaction([site_iA, site_iA4], prefactor_2*S_A*np.array([[0, 1], [0, 0]])),
        Interaction([site_iA4, site_iA], -prefactor_2*S_A*np.array([[0, 1], [0, 0]])),

        Interaction([site_iB, site_iB3], -prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
        Interaction([site_iB3, site_iB], prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
        Interaction([site_iB, site_iB4], -prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
        Interaction([site_iB4, site_iB], prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
        Interaction([site_iB, site_iB2], -prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
        Interaction([site_iB2, site_iB], prefactor_2*S_B*np.array([[0, 1], [0, 0]])),
    ]

    # order S^(1/2)
    prefactor_3 = D*np.sin(theta)*1j/np.sqrt(2)
    caa = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]]])
    aca = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 0]]])
    cca = np.array([[[0, 0], [0, 0]], [[0, 0], [1, 0]]])
    cac = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]])
    expected_magnon_H_3 = [
        # DMI-x -- on-site terms
        Interaction([site_iA, site_iA, site_iA], -1/4*prefactor_3*S_A**(1/2)*(caa - cca)),
        Interaction([site_iA3, site_iA3, site_iA3], 1/4*prefactor_3*S_A**(1/2)*(caa - cca)),
        Interaction([site_iA, site_iA, site_iA], -1/4*prefactor_3*S_A**(1/2)*(caa - cca)),
        Interaction([site_iA1, site_iA1, site_iA1], 1/4*prefactor_3*S_A**(1/2)*(caa - cca)),
        Interaction([site_iA, site_iA, site_iA], -1/4*prefactor_3*S_A**(1/2)*(caa - cca)),
        Interaction([site_iA4, site_iA4, site_iA4], 1/4*prefactor_3*S_A**(1/2)*(caa - cca)),

        Interaction([site_iB, site_iB, site_iB], 1/4*prefactor_3*S_B**(1/2)*(caa - cca)),
        Interaction([site_iB3, site_iB3, site_iB3], -1/4*prefactor_3*S_B**(1/2)*(caa - cca)),
        Interaction([site_iB, site_iB, site_iB], 1/4*prefactor_3*S_B**(1/2)*(caa - cca)),
        Interaction([site_iB4, site_iB4, site_iB4], -1/4*prefactor_3*S_B**(1/2)*(caa - cca)),
        Interaction([site_iB, site_iB, site_iB], 1/4*prefactor_3*S_B**(1/2)*(caa - cca)),
        Interaction([site_iB2, site_iB2, site_iB2], -1/4*prefactor_3*S_B**(1/2)*(caa - cca)),

        # DMI-x -- hopping terms
        Interaction([site_iA, site_iA, site_iA3], prefactor_3*S_A**(1/2)*caa),
        Interaction([site_iA3, site_iA, site_iA], -prefactor_3*S_A**(1/2)*cca),
        Interaction([site_iA, site_iA3, site_iA3], -prefactor_3*S_A**(1/2)*aca),
        Interaction([site_iA3, site_iA3, site_iA], prefactor_3*S_A**(1/2)*cac),
        Interaction([site_iA, site_iA, site_iA1], prefactor_3*S_A**(1/2)*caa),
        Interaction([site_iA1, site_iA, site_iA], -prefactor_3*S_A**(1/2)*cca),
        Interaction([site_iA, site_iA1, site_iA1], -prefactor_3*S_A**(1/2)*aca),
        Interaction([site_iA1, site_iA1, site_iA], prefactor_3*S_A**(1/2)*cac),
        Interaction([site_iA, site_iA, site_iA4], prefactor_3*S_A**(1/2)*caa),
        Interaction([site_iA4, site_iA, site_iA], -prefactor_3*S_A**(1/2)*cca),
        Interaction([site_iA, site_iA4, site_iA4], -prefactor_3*S_A**(1/2)*aca),
        Interaction([site_iA4, site_iA4, site_iA], prefactor_3*S_A**(1/2)*cac),

        Interaction([site_iB, site_iB, site_iB3], -prefactor_3*S_B**(1/2)*caa),
        Interaction([site_iB3, site_iB, site_iB], prefactor_3*S_B**(1/2)*cca),
        Interaction([site_iB, site_iB3, site_iB3], prefactor_3*S_B**(1/2)*aca),
        Interaction([site_iB3, site_iB3, site_iB], -prefactor_3*S_B**(1/2)*cac),
        Interaction([site_iB, site_iB, site_iB4], -prefactor_3*S_B**(1/2)*caa),
        Interaction([site_iB4, site_iB, site_iB], prefactor_3*S_B**(1/2)*cca),
        Interaction([site_iB, site_iB4, site_iB4], prefactor_3*S_B**(1/2)*aca),
        Interaction([site_iB4, site_iB4, site_iB], -prefactor_3*S_B**(1/2)*cac),
        Interaction([site_iB, site_iB, site_iB2], -prefactor_3*S_B**(1/2)*caa),
        Interaction([site_iB2, site_iB, site_iB], prefactor_3*S_B**(1/2)*cca),
        Interaction([site_iB, site_iB2, site_iB2], prefactor_3*S_B**(1/2)*aca),
        Interaction([site_iB2, site_iB2, site_iB], -prefactor_3*S_B**(1/2)*cac),
    ]

    # order S^0
    prefactor_4 = D*np.cos(theta)*1j/4
    caca = np.zeros((2, 2, 2, 2))
    caca[1, 0, 1, 0] = 1.0
    acca = np.zeros((2, 2, 2, 2))
    acca[0, 1, 1, 0] = 1.0
    caac = np.zeros((2, 2, 2, 2))
    caac[1, 0, 0, 1] = 1.0
    expected_magnon_H_4 = [
        # Heisenberg -- on-site terms
        Interaction([site_iA, site_iA, site_iB, site_iB], J*caca),
        Interaction([site_iA, site_iA, site_iB1, site_iB1], J*caca),
        Interaction([site_iA, site_iA, site_iB2, site_iB2], J*caca),

        # Heisenberg -- hopping terms
        Interaction([site_iA, site_iB, site_iB, site_iB], -J/4*(S_A/S_B)**(1/2)*acca),
        Interaction([site_iB, site_iB, site_iB, site_iA], -J/4*(S_A/S_B)**(1/2)*caac),
        Interaction([site_iA, site_iA, site_iA, site_iB], -J/4*(S_B/S_A)**(1/2)*caac),
        Interaction([site_iB, site_iA, site_iA, site_iA], -J/4*(S_B/S_A)**(1/2)*acca),
        Interaction([site_iA, site_iB1, site_iB1, site_iB1], -J/4*(S_A/S_B)**(1/2)*acca),
        Interaction([site_iB1, site_iB1, site_iB1, site_iA], -J/4*(S_A/S_B)**(1/2)*caac),
        Interaction([site_iA, site_iA, site_iA, site_iB1], -J/4*(S_B/S_A)**(1/2)*caac),
        Interaction([site_iB1, site_iA, site_iA, site_iA], -J/4*(S_B/S_A)**(1/2)*acca),
        Interaction([site_iA, site_iB2, site_iB2, site_iB2], -J/4*(S_A/S_B)**(1/2)*acca),
        Interaction([site_iB2, site_iB2, site_iB2, site_iA], -J/4*(S_A/S_B)**(1/2)*caac),
        Interaction([site_iA, site_iA, site_iA, site_iB2], -J/4*(S_B/S_A)**(1/2)*caac),
        Interaction([site_iB2, site_iA, site_iA, site_iA], -J/4*(S_B/S_A)**(1/2)*acca),

        # DMI-z -- hopping terms
        Interaction([site_iA, site_iA3, site_iA3, site_iA3], -prefactor_4*acca),
        Interaction([site_iA3, site_iA3, site_iA3, site_iA], prefactor_4*caac),
        Interaction([site_iA, site_iA, site_iA, site_iA3], -prefactor_4*caac),
        Interaction([site_iA3, site_iA, site_iA, site_iA], prefactor_4*acca),
        Interaction([site_iA, site_iA1, site_iA1, site_iA1], -prefactor_4*acca),
        Interaction([site_iA1, site_iA1, site_iA1, site_iA], prefactor_4*caac),
        Interaction([site_iA, site_iA, site_iA, site_iA1], -prefactor_4*caac),
        Interaction([site_iA1, site_iA, site_iA, site_iA], prefactor_4*acca),
        Interaction([site_iA, site_iA4, site_iA4, site_iA4], -prefactor_4*acca),
        Interaction([site_iA4, site_iA4, site_iA4, site_iA], prefactor_4*caac),
        Interaction([site_iA, site_iA, site_iA, site_iA4], -prefactor_4*caac),
        Interaction([site_iA4, site_iA, site_iA, site_iA], prefactor_4*acca),
        
        Interaction([site_iB, site_iB3, site_iB3, site_iB3], prefactor_4*acca),
        Interaction([site_iB3, site_iB3, site_iB3, site_iB], -prefactor_4*caac),
        Interaction([site_iB, site_iB, site_iB, site_iB3], prefactor_4*caac),
        Interaction([site_iB3, site_iB, site_iB, site_iB], -prefactor_4*acca),
        Interaction([site_iB, site_iB4, site_iB4, site_iB4], prefactor_4*acca),
        Interaction([site_iB4, site_iB4, site_iB4, site_iB], -prefactor_4*caac),
        Interaction([site_iB, site_iB, site_iB, site_iB4], prefactor_4*caac),
        Interaction([site_iB4, site_iB, site_iB, site_iB], -prefactor_4*acca),
        Interaction([site_iB, site_iB2, site_iB2, site_iB2], prefactor_4*acca),
        Interaction([site_iB2, site_iB2, site_iB2, site_iB], -prefactor_4*caac),
        Interaction([site_iB, site_iB, site_iB, site_iB2], prefactor_4*caac),
        Interaction([site_iB2, site_iB, site_iB, site_iB], -prefactor_4*acca),
    ]


    assert_all_real_space_Hamiltonians_equal(model, [
        expected_magnon_H_0, 
        expected_magnon_H_1,
        expected_magnon_H_2, 
        expected_magnon_H_3, 
        expected_magnon_H_4,
    ])