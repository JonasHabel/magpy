import numpy as np
from magpy.models import Model
from magpy.largeS import LSWT
from magpy.momenta_utils import Momenta
from magpy.correlators import magnon_correlators
from . import test_models


def test_correlators_FM_Heisenberg_chain():
    model, _ = test_models.FM_Heisenberg_chain()

    N_BZ = 10,
    ks_BZ = Momenta.of_BZ(model.lattice, N_BZ)
    _, eigvs_BZ = LSWT.get_eigensystems_momentum_space(model, ks_BZ)
    
    # MOMENTUM SPACE
    correlators_mom_space = magnon_correlators.compute_momentum_space_correlators_LSWT(eigvs_BZ, strip=True).raw_quantity

    assert correlators_mom_space.shape == (*N_BZ, 2, 2)
    assert np.allclose(
        correlators_mom_space, 
        np.tile(np.array([[0, 1], [0, 0]]), (*N_BZ, 1, 1)),
    )

    # REAL SPACE
    N_sites = N_BZ
    bravais_coords = model.lattice.sample_Bravais_lattice_in_Bravais_coords(N_sites)
    correlators_real_space = \
        magnon_correlators.compute_real_space_correlator_LSWT(
            ks_BZ, eigvs_BZ, bravais_coords
        )
    
    expected_correlators_real_space = np.zeros((*N_sites, 2, 2))
    expected_correlators_real_space[0, 0, 1] = 1.
    assert correlators_real_space.shape == (*N_sites, 2, 2)
    assert np.allclose(
        correlators_real_space,
        expected_correlators_real_space,
    )




def test_correlators_AFM_Heisenberg_chain():
    model, params = test_models.AFM_Heisenberg_chain()
    J, S_A, S_B = params

    N_BZ = 10,
    ks_BZ = Momenta.of_BZ(model.lattice, N_BZ)
    _, eigvs_BZ = LSWT.get_eigensystems_momentum_space(model, ks_BZ)
    
    # MOMENTUM SPACE
    correlators_mom_space = magnon_correlators.compute_momentum_space_correlators_LSWT(eigvs_BZ, strip=True).raw_quantity

    # BdG Hamiltonian of the form [[epsilon, gamma*], [gamma, epsilon]]
    epsilon = J*(S_A + S_B)
    gamma = lambda k: -J*np.sqrt(S_A*S_B) * (1 + np.exp(1j*k[0]))
    # BdG transformation of the form U_k = [[u_k, v_k], [v_k*, u_k*]]
    # where u_k = cosh(beta) and v_k = e^(i*phi)*sinh(beta) (up to global phase)
    beta = lambda k: 0.5 * np.arctanh(-np.abs(gamma(k)) / epsilon)
    phi = lambda k: -np.angle(gamma(k))
    BdG_coeffs_u_k, BdG_coeffs_v_k = np.array([
        [np.cosh(beta(k)), np.exp(1j*phi(k)) * np.sinh(beta(k))]
        for k in ks_BZ.k_arrays[0]
    ]).T
    BdG_coeffs_u_minus_k, BdG_coeffs_v_minus_k = np.array([
        [np.cosh(beta(-k)), np.exp(1j*phi(-k)) * np.sinh(beta(-k))]
        for k in ks_BZ.k_arrays[0]
    ]).T

    assert correlators_mom_space.shape == (*N_BZ, 4, 4)
    assert np.allclose(
        correlators_mom_space, 
        np.array([
            [
                [0,                          u_k*u_k.conj(),        u_k*v_k,        0],                                  # <aa>   <aa^†>,  <ab>   <ab^†>
                [v_minus_k*v_minus_k.conj(), 0,                     0,              u_minus_k.conj()*v_minus_k.conj()],  # <a^†a> <a^†a^†> <a^†b> <a^†b^†>
                [u_minus_k*v_minus_k,        0,                     0,              u_minus_k*u_minus_k.conj()],         # <ba>   <ba^†>   <bb>   <bb^†>
                [0,                          v_k.conj()*u_k.conj(), v_k*v_k.conj(), 0],                                  # <b^†a> <b^†a^†> <b^†b> <b^†b^†>
            ] for u_k, u_minus_k, v_k, v_minus_k in zip(
                BdG_coeffs_u_k, BdG_coeffs_u_minus_k, 
                BdG_coeffs_v_k, BdG_coeffs_v_minus_k
            )
        ]),
    )
    
    # REAL SPACE
    N_sites = N_BZ
    bravais_coords = model.lattice.sample_Bravais_lattice_in_Bravais_coords(N_sites)
    correlators_real_space = \
        magnon_correlators.compute_real_space_correlator_LSWT(
            ks_BZ, eigvs_BZ, bravais_coords
        )
    
    # expected_correlators_real_space = np.zeros((*N_sites, 4, 4))
    # expected_correlators_real_space[0] = np.array([
    #     [0, ..., ..., 0],
    #     [..., 0, 0, ...],
    #     [..., 0, 0, ...],
    #     [0, ..., ..., 0],
    # ])
    assert correlators_real_space.shape == (*N_sites, 4, 4)
    # assert np.allclose(
    #     correlators_real_space,
    #     expected_correlators_real_space
    # )
