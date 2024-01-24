import numpy as np
from magpy.largeS import LSWT
from magpy.lattice import BravaisLattice
from magpy.interactions import Interaction
from . import test_models


def assert_LSWT_BdG_Hamiltonians_equal(model, k, expected_H_LSWT_BdG):
    H_LSWT_BdG = LSWT.compute_LSWT_Hamiltonian_momentum_space_BdG(model, k)
    assert np.allclose(H_LSWT_BdG, expected_H_LSWT_BdG)


def assert_LSWT_eigensystems_equal(model, k, expected_eigw, expected_eigv):
    eigw, eigv = LSWT.get_eigensystem_momentum_space(model, k)
    assert np.allclose(eigw, expected_eigw)
    eigvs_allclose_up_to_phase(eigv, expected_eigv)


def eigvs_allclose_up_to_phase(eigv, expected_eigv):
    for n in range(len(eigv[0])):
        assert np.allclose(
            np.abs(eigv[:,n].T.conj() @ expected_eigv[:,n]), 
            np.abs(eigv[:,n].T.conj() @ eigv[:,n]),
        )



def test_momentum_space_Hamiltonian_AFM_Heisenberg_chain():
    model, (J, S_A, S_B) = test_models.AFM_Heisenberg_chain()

    np.random.seed(1)
    q = np.random.rand(1)

    B = -(1 + np.exp(1j*q[0]))*np.sqrt(S_A*S_B)
    expected_H_LSWT_BdG = J*np.array([
        [2*S_B, 0, 0, np.conj(B)],
        [0, 2*S_B, np.conj(B), 0],
        [0, B, 2*S_A, 0],
        [B, 0, 0, 2*S_A],
    ])
    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)

    expected_eigw = np.array([
        J*(S_B - S_A + np.sqrt(S_A**2 + S_B**2 - 2*S_A*S_B*np.cos(q[0]))),
        -J*(S_B - S_A + np.sqrt(S_A**2 + S_B**2 - 2*S_A*S_B*np.cos(q[0]))),
        -J*(S_B - S_A - np.sqrt(S_A**2 + S_B**2 - 2*S_A*S_B*np.cos(q[0]))),
        J*(S_B - S_A - np.sqrt(S_A**2 + S_B**2 - 2*S_A*S_B*np.cos(q[0]))),
    ])
    tanh_u = 1/np.abs(B) * (-(S_A + S_B) + np.sqrt((S_A + S_B)**2 - np.abs(B)**2))
    cosh, sinh = lambda tanh: 1/np.sqrt(1 - tanh**2), lambda tanh: tanh/np.sqrt(1 - tanh**2)
    phase = B / np.abs(B)
    half_of_eigv = np.array([
        [cosh(tanh_u), sinh(tanh_u)],
        [phase*sinh(tanh_u), phase*cosh(tanh_u)],
    ])
    expected_eigv = np.zeros((4, 4), dtype=np.complex128)
    expected_eigv[1:3, 1:3] = half_of_eigv
    expected_eigv[::3, ::3] = half_of_eigv
    assert_LSWT_eigensystems_equal(model, q, expected_eigw, expected_eigv)



def test_momentum_space_Hamiltonian_FM_Heisenberg_chain():
    model, (J, S) = test_models.FM_Heisenberg_chain()

    delta_ij = np.array([1])

    np.random.seed(1)
    q = np.random.rand(1)

    expected_H_LSWT_BdG = 2*J*S*np.diag([-1 + np.cos(q.dot(delta_ij))] * 2)
    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)

    expected_eigw = 2*J*S*(-1 + np.cos(q)) * np.array([1, -1])
    expected_eigv = np.eye(2)
    assert_LSWT_eigensystems_equal(model, q, expected_eigw, expected_eigv)




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

    beta_1 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nn)) for nn in nns]))
    beta_2 = lambda K: np.sum(np.array([np.exp(1j*K.dot(nnn)) for nnn in nnns]))
    a = lambda K: J*beta_1(K)*np.sqrt(S_A*S_B)
    b = lambda K: 1j*D*np.cos(theta)*(beta_2(K) - beta_2(-K))
    expected_H_LSWT_BdG = np.array([
        [-3*J*S_B + b(-q)*S_A, 0, a(q), 0],
        [0, -3*J*S_B + b(q)*S_A, 0, a(q)],
        [a(-q), 0, -3*J*S_A - b(-q)*S_B, 0],
        [0, a(-q), 0, -3*J*S_A - b(q)*S_B],
    ])
    assert_LSWT_BdG_Hamiltonians_equal(model, q, expected_H_LSWT_BdG)

    expected_eigw = np.array([
        1/2*((-3*J*(S_B + S_A) + b(-q)*(S_A - S_B)) - np.sqrt((-3*J*(S_B - S_A) + b(-q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2)),
        -1/2*((-3*J*(S_B + S_A) + b(q)*(S_A - S_B)) - np.sqrt((-3*J*(S_B - S_A) + b(q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2)),
        1/2*((-3*J*(S_B + S_A) + b(-q)*(S_A - S_B)) + np.sqrt((-3*J*(S_B - S_A) + b(-q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2)),
        -1/2*((-3*J*(S_B + S_A) + b(q)*(S_A - S_B)) + np.sqrt((-3*J*(S_B - S_A) + b(q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2)),
    ])
    tan_u1 = 1/(2*np.abs(a(q))) * (-(-3*J*(S_B - S_A) + b(-q)*(S_A + S_B)) + np.sqrt((-3*J*(S_B - S_A) - b(-q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2))
    tan_u2 = 1/(2*np.abs(a(q))) * (-(-3*J*(S_B - S_A) + b(q)*(S_A + S_B)) + np.sqrt((-3*J*(S_B - S_A) - b(q)*(S_A + S_B))**2 + 4*np.abs(a(q))**2))
    cos, sin = lambda tan: 1/np.sqrt(1 + tan**2), lambda tan: tan/np.sqrt(1 + tan**2)
    phase = a(-q) / np.abs(a(q))
    half_of_eigv = lambda tan_u: np.array([
        [-sin(tan_u), cos(tan_u)],
        [phase*cos(tan_u), phase*sin(tan_u)],
    ])
    expected_eigv = np.zeros((4, 4), dtype=np.complex128)
    expected_eigv[::2, ::2] = half_of_eigv(tan_u1)
    expected_eigv[1::2, 1::2] = half_of_eigv(tan_u2)
    assert_LSWT_eigensystems_equal(model, q, expected_eigw, expected_eigv)

    