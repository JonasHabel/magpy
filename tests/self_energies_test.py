from magpy.lattice import *
from magpy import models
from magpy.interactions import *
from magpy.largeS import real_space, momentum_space, eigenspace
from magpy.largeS import LSWT
from magpy import self_energies
import numpy as np




def test_two_site_quantum_dot_with_DMI():
    latt = DotLattice(2)
    mod = models.Model(latt, [
        DMInteraction(BravaisLattice.Edge(np.array([]), [0, 1]), D=np.array([1.0, 0, 0]))
    ], classical_ground_state=np.array([[0, 0, 1], [0, 0, 1]]))

    # VERTICES REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)

    # VERTICES MOMENTUM SPACE
    verts_mom_space = momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
        mod, np.zeros((2, 1)), verts_real_space)

    # VERTICES EIGENSPACE
    _, eigvs = LSWT.get_eigensystem_momentum_space(mod, np.zeros((0, 1)))
    eigvs = np.array([eigvs, eigvs, eigvs])
    verts_eigenspace = eigenspace.compute_magnon_Hamiltonian_with_permutations(
        eigvs, verts_mom_space)

    # NORMAL-ORDER AND SYMMETRIZE
    verts_eigspace_nosym = eigenspace.normal_order_and_symmetrize_magnon_Hamiltonian(verts_eigenspace)

    # SELF-ENERGY
    freqs = np.array([0.0]) # np.linspace(0, 5, 11)
    reg = 0.05
    expected_se_pp = np.array([
        0.5/(freq + 1j*reg) * np.array([
            [9./8, -1.],
            [-1., 9./8],
        ]) for freq in freqs
    ])
    se_p_pp_p = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, np.array([0.0, 0, 0, 0]), np.array([0.0, 0, 0, 0]),
        verts_eigspace_nosym, 0.0, reg)
    
    assert np.allclose(se_p_pp_p, expected_se_pp)


def test_field_orthogonal_to_quantization_direction():
    latt = SquareLattice()
    B = np.array([1, 2, 3])
    inter = [
        MagneticField(latt, 0, B),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))

    # VERTICES REAL SPACE
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)

    # VERTICES MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_for_loop_momentum = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ)
    
    # VERTICES EIGENSPACE
    _, eigvs_at_k = LSWT.get_eigensystem_momentum_space(mod, k)
    energies_BZ, eigvs_BZ = \
        LSWT.get_eigensystem_for_Brillouin_zone(mod, N_BZ)
    _, eigvs_minus_k_minus_BZ = \
        LSWT.get_eigensystem_for_loop_momentum(mod, -k, N_BZ)
    verts_eigenspace_for_loop_momentum = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ, 
            verts_for_loop_momentum)
    
    # SELF-ENERGY
    # TODO FIX ENERGIES OF LOOP MOMENTA!!!
    freqs = np.linspace(0, 5, 11)
    energies_k_minus_BZ, _ = LSWT.get_eigensystem_for_loop_momentum(mod, k, N_BZ)
    reg = 0.05
    T = 0.1
    B_xy_sq = B[0]**2 + B[1]**2
    # def n_B(E, T): return 1/(np.exp(E/T) - 1)
    # expected_se_pp = np.array(1/16 * (n_B(B[2], T) - n_B(-B[2], T)) * B_xy_sq / (freqs - 2*B[2] + 1j*reg)) \
    #     .reshape((len(freqs), 1, 1))
    # expected_se_ph = np.array(-1/16 * (n_B(B[2], T) - n_B(B[2], T)) * B_xy_sq / (freqs + 1j*reg)) \
    #     .reshape((len(freqs), 1, 1))
    # expected_se_hh = np.array(1/16 * (n_B(-B[2], T) - n_B(B[2], T)) * B_xy_sq / (freqs + 2*B[2] + 1j*reg)) \
    #     .reshape((len(freqs), 1, 1))
    expected_se_pp = np.array(1/16 * B_xy_sq / (freqs - 2*B[2] + 1j*reg)) \
        .reshape((len(freqs), 1, 1))
    expected_se_hh = np.array(1/16 * B_xy_sq / (freqs + 2*B[2] + 1j*reg)) \
        .reshape((len(freqs), 1, 1))
    
    se_p_pp_p = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "pp", "p"], reg)
    assert np.allclose(se_p_pp_p, expected_se_pp)
    
    se_p_ph_p = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "ph", "p"], reg)
    assert np.allclose(se_p_ph_p, np.zeros(len(freqs)))
    
    se_p_hh_p = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "hh", "p"], reg)
    assert np.allclose(se_p_hh_p, np.zeros(len(freqs)))
    
    se_p_pp_h = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "pp", "h"], reg)
    assert np.allclose(se_p_pp_h, np.zeros(len(freqs)))
    
    se_p_ph_h = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "ph", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_p_ph_h = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["p", "hh", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_h_hh_h = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, T, ["h", "hh", "h"], reg)
    assert np.allclose(se_h_hh_h, expected_se_hh)



def test_1D_BdG_chain_with_cubic_interaction():
    latt = ChainLattice(1)
    mod = Model(latt, [
        Interaction([
            BravaisLattice.Site(np.array([0]), 0),
            BravaisLattice.Site(np.array([1]), 0),
        ], np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])),
        MagneticField(latt, sublattice_index=0, B=np.array([0, 0, 3.0]))
    ], np.array([[0, 0, 0.5]]))

    H_LSWT_real_space = LSWT.compute_LSWT_Hamiltonian_real_space(mod)
    assert np.allclose(
        H_LSWT_real_space[0].interaction_tensor,
        -0.5j*np.array([[1, 0], [0, -1]]))
    assert np.allclose(
        H_LSWT_real_space[3].interaction_tensor,
        np.array([[0, 0], [3.0, 0]]))
    
    num_ks = 10
    kpath = ReciprocalLattice.MomentumPath(
        np.linspace(0, 2*np.pi, num_ks, endpoint=False).reshape(num_ks, 1))
    sigma_y, sigma_z = np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    h0, Delta = 3.0, lambda k: -np.cos(k[0])
    H_LSWT_along_kpath = LSWT.compute_LSWT_Hamiltonian_along_momentum_path(
        mod, kpath)
    expected_H_LSWT_along_kpath = np.array([
        h0*np.eye(2) + Delta(k)*sigma_y \
        for k in kpath.ks
    ], dtype=complex)
    assert np.allclose(H_LSWT_along_kpath, expected_H_LSWT_along_kpath)
    
    eigws, eigvs = LSWT.get_eigensystem_along_momentum_path(mod, kpath)
    epsilon = lambda k: np.sqrt(h0**2 - np.abs(Delta(k))**2)
    expected_eigws = np.array([
        epsilon(k) * np.array([1, -1]) \
        for k in kpath.ks
    ], dtype=float)
    assert np.allclose(eigws, expected_eigws)
    # check orthonormality wrt bogo metric
    assert np.allclose(
        np.einsum("kin,ij,kjm->knm", eigvs.conj(), sigma_z, eigvs),
        np.array([sigma_z] * num_ks)) 
    # check if diagonalizes Hamiltonian
    assert np.allclose(
        np.einsum("kin,kij,kjm->knm", eigvs.conj(), H_LSWT_along_kpath, eigvs),
        np.array([np.diag([eigw[0], np.abs(eigw[1])]) for eigw in eigws])) 
    # check equality with expected values
    u = lambda k: 1/(2*np.sqrt(epsilon(k))) * (np.sqrt(h0 - Delta(k)) + np.sqrt(h0 + Delta(k)))
    v = lambda k: 1/(2*np.sqrt(epsilon(k))) * (np.sqrt(h0 - Delta(k)) - np.sqrt(h0 + Delta(k)))
    expected_eigvs = np.array([
        u(k)*np.eye(2) + v(k)*sigma_y \
        for k in kpath.ks
    ])
    assert np.allclose(eigvs, expected_eigvs)

    # make up some real space interaction vertices
    cubic_verts_real_space = [
        Interaction([
            BravaisLattice.Site(np.array([0]), 0),
            BravaisLattice.Site(np.array([0]), 0),
            BravaisLattice.Site(np.array([1]), 0),
        ], np.array([
            [[0, 0], [0, 0]],
            [[1, 0], [0, 0]],
        ])),
        Interaction([
            BravaisLattice.Site(np.array([0]), 0),
            BravaisLattice.Site(np.array([1]), 0),
            BravaisLattice.Site(np.array([0]), 0),
        ], np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [1, 0]],
        ]))
    ]

    # check momentum space interaction vertices
    k_idx = 2
    k = kpath.ks[k_idx]
    momenta_BZ = kpath.ks
    cubic_verts_for_loop_momentum_k = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ, cubic_verts_real_space)
    caa = np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [0, 0]],
    ])
    cca = np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ])
    exp = np.exp
    expected_cubic_verts_for_loop_momentum = np.array([
        [exp(1j*k[0])*caa + exp(1j*p[0])*cca for p in momenta_BZ],
        [exp(1j*p[0])*caa + exp(1j*k[0])*cca for p in momenta_BZ],
        [exp(1j*k[0])*caa + exp(-1j*(k[0]+p[0]))*cca for p in momenta_BZ],
        [exp(-1j*(k[0]+p[0]))*caa + exp(1j*k[0])*cca for p in momenta_BZ],
        [exp(1j*p[0])*caa + exp(-1j*(k[0]+p[0]))*cca for p in momenta_BZ],
        [exp(-1j*(k[0]+p[0]))*caa + exp(1j*p[0])*cca for p in momenta_BZ],
    ])
    assert np.allclose(
        cubic_verts_for_loop_momentum_k,
        expected_cubic_verts_for_loop_momentum)
    
    # check LSWT eigenspace interaction vertices
    eigvs_minus_k_minus_BZ = np.array([
        eigvs[(-k_idx-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    cubic_verts_eigenspace_for_loop_momentum_k = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs[k_idx], eigvs, eigvs_minus_k_minus_BZ,
            cubic_verts_for_loop_momentum_k
        )
    expected_cubic_verts_eigenspace_for_loop_momentum_k = np.array([
        # -k-p, p, k
        [[
            [
                [
                    v(-k-p)*u(k)*(exp(1j*k[0])*1j*u(p) - exp(1j*p[0])*v(p)),
                    v(-k-p)*v(k)*(exp(1j*k[0])*u(p) + exp(1j*p[0])*1j*v(p))
                ], [
                    v(-k-p)*u(k)*(exp(1j*k[0])*v(p) + exp(1j*p[0])*1j*u(p)),
                    v(-k-p)*v(k)*(-exp(1j*k[0])*1j*v(p) + exp(1j*p[0])*u(p)),
                ],
            ], [
                [
                    u(-k-p)*u(k)*(exp(1j*k[0])*u(p) + exp(1j*p[0])*1j*v(p)),
                    u(-k-p)*v(k)*(-exp(1j*k[0])*1j*u(p) + exp(1j*p[0])*v(p)),
                ], [
                    u(-k-p)*u(k)*(-exp(1j*k[0])*1j*v(p) + exp(1j*p[0])*u(p)),
                    u(-k-p)*v(k)*(-exp(1j*k[0])*v(p) - exp(1j*p[0])*1j*u(p))
                ],
            ],
        ] for p in momenta_BZ],
        # -k-p, k, p
        [[
            [
                [
                    v(-k-p)*u(p)*(exp(1j*p[0])*1j*u(k) - exp(1j*k[0])*v(k)),
                    v(-k-p)*v(p)*(exp(1j*p[0])*u(k) + exp(1j*k[0])*1j*v(k))
                ], [
                    v(-k-p)*u(p)*(exp(1j*p[0])*v(k) + exp(1j*k[0])*1j*u(k)),
                    v(-k-p)*v(p)*(-exp(1j*p[0])*1j*v(k) + exp(1j*k[0])*u(k)),
                ],
            ], [
                [
                    u(-k-p)*u(p)*(exp(1j*p[0])*u(k) + exp(1j*k[0])*1j*v(k)),
                    u(-k-p)*v(p)*(-exp(1j*p[0])*1j*u(k) + exp(1j*k[0])*v(k)),
                ], [
                    u(-k-p)*u(p)*(-exp(1j*p[0])*1j*v(k) + exp(1j*k[0])*u(k)),
                    u(-k-p)*v(p)*(-exp(1j*p[0])*v(k) - exp(1j*k[0])*1j*u(k))
                ],
            ],
        ] for p in momenta_BZ],
        # p, -k-p, k
        [[
            [
                [
                    v(p)*u(k)*(exp(1j*k[0])*1j*u(-k-p) - exp(-1j*(k[0]+p[0]))*v(-k-p)),
                    v(p)*v(k)*(exp(1j*k[0])*u(-k-p) + exp(-1j*(k[0]+p[0]))*1j*v(-k-p))
                ], [
                    v(p)*u(k)*(exp(1j*k[0])*v(-k-p) + exp(-1j*(k[0]+p[0]))*1j*u(-k-p)),
                    v(p)*v(k)*(-exp(1j*k[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)),
                ],
            ], [
                [
                    u(p)*u(k)*(exp(1j*k[0])*u(-k-p) + exp(-1j*(k[0]+p[0]))*1j*v(-k-p)),
                    u(p)*v(k)*(-exp(1j*k[0])*1j*u(-k-p) + exp(-1j*(k[0]+p[0]))*v(-k-p)),
                ], [
                    u(p)*u(k)*(-exp(1j*k[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)),
                    u(p)*v(k)*(-exp(1j*k[0])*v(-k-p) - exp(-1j*(k[0]+p[0]))*1j*u(-k-p))
                ],
            ],
        ] for p in momenta_BZ],
        # p, k, -k-p
        [[
            [
                [
                    v(p)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*1j*u(k) - exp(1j*k[0])*v(k)),
                    v(p)*v(-k-p)*(exp(-1j*(k[0]+p[0]))*u(k) + exp(1j*k[0])*1j*v(k))
                ], [
                    v(p)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*v(k) + exp(1j*k[0])*1j*u(k)),
                    v(p)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*v(k) + exp(1j*k[0])*u(k)),
                ],
            ], [
                [
                    u(p)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*u(k) + exp(1j*k[0])*1j*v(k)),
                    u(p)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*u(k) + exp(1j*k[0])*v(k)),
                ], [
                    u(p)*u(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*v(k) + exp(1j*k[0])*u(k)),
                    u(p)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*v(k) - exp(1j*k[0])*1j*u(k))
                ],
            ],
        ] for p in momenta_BZ],
        # k, -k-p, p
        [[
            [
                [
                    v(k)*u(p)*(exp(1j*p[0])*1j*u(-k-p) - exp(-1j*(k[0]+p[0]))*v(-k-p)),
                    v(k)*v(p)*(exp(1j*p[0])*u(-k-p) + exp(-1j*(k[0]+p[0]))*1j*v(-k-p))
                ], [
                    v(k)*u(p)*(exp(1j*p[0])*v(-k-p) + exp(-1j*(k[0]+p[0]))*1j*u(-k-p)),
                    v(k)*v(p)*(-exp(1j*p[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)),
                ],
            ], [
                [
                    u(k)*u(p)*(exp(1j*p[0])*u(-k-p) + exp(-1j*(k[0]+p[0]))*1j*v(-k-p)),
                    u(k)*v(p)*(-exp(1j*p[0])*1j*u(-k-p) + exp(-1j*(k[0]+p[0]))*v(-k-p)),
                ], [
                    u(k)*u(p)*(-exp(1j*p[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)),
                    u(k)*v(p)*(-exp(1j*p[0])*v(-k-p) - exp(-1j*(k[0]+p[0]))*1j*u(-k-p))
                ],
            ],
        ] for p in momenta_BZ],
        # k, p, -k-p
        [[
            [
                [
                    v(k)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*1j*u(p) - exp(1j*p[0])*v(p)),
                    v(k)*v(-k-p)*(exp(-1j*(k[0]+p[0]))*u(p) + exp(1j*p[0])*1j*v(p))
                ], [
                    v(k)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*v(p) + exp(1j*p[0])*1j*u(p)),
                    v(k)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*v(p) + exp(1j*p[0])*u(p)),
                ],
            ], [
                [
                    u(k)*u(-k-p)*(exp(-1j*(k[0]+p[0]))*u(p) + exp(1j*p[0])*1j*v(p)),
                    u(k)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*u(p) + exp(1j*p[0])*v(p)),
                ], [
                    u(k)*u(-k-p)*(-exp(-1j*(k[0]+p[0]))*1j*v(p) + exp(1j*p[0])*u(p)),
                    u(k)*v(-k-p)*(-exp(-1j*(k[0]+p[0]))*v(p) - exp(1j*p[0])*1j*u(p))
                ],
            ],
        ] for p in momenta_BZ],
    ])
    assert np.allclose(
        cubic_verts_eigenspace_for_loop_momentum_k,
        expected_cubic_verts_eigenspace_for_loop_momentum_k)
    
    # check all Wick contractions of the pp-bubble, encapsulated in the
    # creator-creator-annihilator (cca) symmetrized interaction vertex
    freqs = np.linspace(0, 5, 5, endpoint=False)
    eigws_minus_k_minus_BZ = np.array([
        eigws[(-k_idx-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    particle_hole_states = [["p"], ["p", "p"], ["p"]]
    from magpy import self_energies_old
    symmetrized_vertices_cca_k = \
        self_energies_old.__get_all_Wick_contractions_of_cubic_vertices_for_loop_momentum(
            expected_cubic_verts_eigenspace_for_loop_momentum_k,
            np.array([[1, 1, 0], [1, 0, 0]]), (num_ks,))
    expected_symmetrized_vertices_cca_k = np.array([
        v(k)*v(-k-p) * (-exp(-1j*(k[0]+p[0]))*1j*v(p) + exp(1j*p[0])*u(p)) + \
        v(k)*v(p) * (-exp(1j*p[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)) + \
        u(p)*v(-k-p) * (-exp(-1j*(k[0]+p[0]))*1j*u(k) + exp(1j*k[0])*v(k)) + \
        u(-k-p)*v(p) * (-exp(1j*p[0])*1j*u(k) + exp(1j*k[0])*v(k)) + \
        u(-k-p)*u(k) * (-exp(1j*k[0])*1j*v(p) + exp(1j*p[0])*u(p)) + \
        u(p)*u(k) * (-exp(1j*k[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)) \
        for p in momenta_BZ
    ])
    expected_symmetrized_vertices_cca_k_2 = np.array([
        cubic_verts_eigenspace_for_loop_momentum_k[0, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace_for_loop_momentum_k[1, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[2, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace_for_loop_momentum_k[3, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[4, p_idx, 0, 1, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[5, p_idx, 0, 1, 1] \
        for p_idx, p in enumerate(momenta_BZ)
    ])
    assert np.allclose(
        expected_symmetrized_vertices_cca_k,
        expected_symmetrized_vertices_cca_k_2)
    assert np.allclose(
        symmetrized_vertices_cca_k[1],
        symmetrized_vertices_cca_k[0].conj())
    assert np.allclose(
        symmetrized_vertices_cca_k[0],
        expected_symmetrized_vertices_cca_k.reshape(num_ks, 1, 1, 1))

    # check invariance of symmetrized cca interaction vertex under swapping
    # the sign of the loop momentum (p -> -p)
    eigvs_minus_k_plus_BZ = np.array([
        eigvs[(-k_idx+p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    eigvs_minus_BZ = np.array([
        eigvs[(-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    cubic_verts_for_loop_momentum_swapped_k = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, -momenta_BZ, cubic_verts_real_space)
    cubic_verts_eigenspace_for_loop_momentum_swapped_k = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs[k_idx], eigvs_minus_BZ, eigvs_minus_k_plus_BZ,
            cubic_verts_for_loop_momentum_swapped_k
        )
    symmetrized_vertices_swapped_cca_k = \
        self_energies_old.__get_all_Wick_contractions_of_cubic_vertices_for_loop_momentum(
            cubic_verts_eigenspace_for_loop_momentum_swapped_k,
            np.array([[1, 1, 0], [1, 0, 0]]), (num_ks,))
    expected_symmetrized_vertices_swapped_cca_k = np.array([
        expected_symmetrized_vertices_cca_k[(-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    assert np.allclose(
        symmetrized_vertices_swapped_cca_k[0],
        expected_symmetrized_vertices_swapped_cca_k.reshape(num_ks, 1, 1, 1))
    
    # check if the annihilator-annihilator-creator (aac) symmetrized interaction
    # vertex maps to the conjugate of the caa vertex with inverted momenta
    eigvs_k_plus_BZ = np.array([
        eigvs[(k_idx+p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    cubic_verts_for_loop_momentum_minusk = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, -k, -momenta_BZ, cubic_verts_real_space)
    cubic_verts_eigenspace_for_loop_momentum_minusk = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs[(-k_idx) % num_ks], eigvs_minus_BZ, eigvs_k_plus_BZ,
            cubic_verts_for_loop_momentum_minusk
        )
    expected_cubic_verts_eigenspace_for_loop_momentum_minusk = np.array([
        # k+p, -p, -k
        [[
            [
                [
                    v(k+p)*u(-k)*(exp(-1j*k[0])*1j*u(-p) - exp(-1j*p[0])*v(-p)),
                    v(k+p)*v(-k)*(exp(-1j*k[0])*u(-p) + exp(-1j*p[0])*1j*v(-p))
                ], [
                    v(k+p)*u(-k)*(exp(-1j*k[0])*v(-p) + exp(-1j*p[0])*1j*u(-p)),
                    v(k+p)*v(-k)*(-exp(-1j*k[0])*1j*v(-p) + exp(-1j*p[0])*u(-p)),
                ],
            ], [
                [
                    u(k+p)*u(-k)*(exp(-1j*k[0])*u(-p) + exp(-1j*p[0])*1j*v(-p)),
                    u(k+p)*v(-k)*(-exp(-1j*k[0])*1j*u(-p) + exp(-1j*p[0])*v(-p)),
                ], [
                    u(k+p)*u(-k)*(-exp(-1j*k[0])*1j*v(-p) + exp(-1j*p[0])*u(-p)),
                    u(k+p)*v(-k)*(-exp(-1j*k[0])*v(-p) - exp(-1j*p[0])*1j*u(-p))
                ],
            ],
        ] for p in momenta_BZ],
        # k+p, -k, -p
        [[
            [
                [
                    v(k+p)*u(-p)*(exp(-1j*p[0])*1j*u(-k) - exp(-1j*k[0])*v(-k)),
                    v(k+p)*v(-p)*(exp(-1j*p[0])*u(-k) + exp(-1j*k[0])*1j*v(-k))
                ], [
                    v(k+p)*u(-p)*(exp(-1j*p[0])*v(-k) + exp(-1j*k[0])*1j*u(-k)),
                    v(k+p)*v(-p)*(-exp(-1j*p[0])*1j*v(-k) + exp(-1j*k[0])*u(-k)),
                ],
            ], [
                [
                    u(k+p)*u(-p)*(exp(-1j*p[0])*u(-k) + exp(-1j*k[0])*1j*v(-k)),
                    u(k+p)*v(-p)*(-exp(-1j*p[0])*1j*u(-k) + exp(-1j*k[0])*v(-k)),
                ], [
                    u(k+p)*u(-p)*(-exp(-1j*p[0])*1j*v(-k) + exp(-1j*k[0])*u(-k)),
                    u(k+p)*v(-p)*(-exp(-1j*p[0])*v(-k) - exp(-1j*k[0])*1j*u(-k))
                ],
            ],
        ] for p in momenta_BZ],
        # -p, k+p, -k
        [[
            [
                [
                    v(-p)*u(-k)*(exp(-1j*k[0])*1j*u(k+p) - exp(1j*(k[0]+p[0]))*v(k+p)),
                    v(-p)*v(-k)*(exp(-1j*k[0])*u(k+p) + exp(1j*(k[0]+p[0]))*1j*v(k+p))
                ], [
                    v(-p)*u(-k)*(exp(-1j*k[0])*v(k+p) + exp(1j*(k[0]+p[0]))*1j*u(k+p)),
                    v(-p)*v(-k)*(-exp(-1j*k[0])*1j*v(k+p) + exp(1j*(k[0]+p[0]))*u(k+p)),
                ],
            ], [
                [
                    u(-p)*u(-k)*(exp(-1j*k[0])*u(k+p) + exp(1j*(k[0]+p[0]))*1j*v(k+p)),
                    u(-p)*v(-k)*(-exp(-1j*k[0])*1j*u(k+p) + exp(1j*(k[0]+p[0]))*v(k+p)),
                ], [
                    u(-p)*u(-k)*(-exp(-1j*k[0])*1j*v(k+p) + exp(1j*(k[0]+p[0]))*u(k+p)),
                    u(-p)*v(-k)*(-exp(-1j*k[0])*v(k+p) - exp(1j*(k[0]+p[0]))*1j*u(k+p))
                ],
            ],
        ] for p in momenta_BZ],
        # -p, -k, k+p
        [[
            [
                [
                    v(-p)*u(k+p)*(exp(1j*(k[0]+p[0]))*1j*u(-k) - exp(-1j*k[0])*v(-k)),
                    v(-p)*v(k+p)*(exp(1j*(k[0]+p[0]))*u(-k) + exp(-1j*k[0])*1j*v(-k))
                ], [
                    v(-p)*u(k+p)*(exp(1j*(k[0]+p[0]))*v(-k) + exp(-1j*k[0])*1j*u(-k)),
                    v(-p)*v(k+p)*(-exp(1j*(k[0]+p[0]))*1j*v(-k) + exp(-1j*k[0])*u(-k)),
                ],
            ], [
                [
                    u(-p)*u(k+p)*(exp(1j*(k[0]+p[0]))*u(-k) + exp(-1j*k[0])*1j*v(-k)),
                    u(-p)*v(k+p)*(-exp(1j*(k[0]+p[0]))*1j*u(-k) + exp(-1j*k[0])*v(-k)),
                ], [
                    u(-p)*u(k+p)*(-exp(1j*(k[0]+p[0]))*1j*v(-k) + exp(-1j*k[0])*u(-k)),
                    u(-p)*v(k+p)*(-exp(1j*(k[0]+p[0]))*v(-k) - exp(-1j*k[0])*1j*u(-k))
                ],
            ],
        ] for p in momenta_BZ],
        # -k, k-p, -p
        [[
            [
                [
                    v(-k)*u(-p)*(exp(-1j*p[0])*1j*u(k+p) - exp(1j*(k[0]+p[0]))*v(k+p)),
                    v(-k)*v(-p)*(exp(-1j*p[0])*u(k+p) + exp(1j*(k[0]+p[0]))*1j*v(k+p))
                ], [
                    v(-k)*u(-p)*(exp(-1j*p[0])*v(k+p) + exp(1j*(k[0]+p[0]))*1j*u(k+p)),
                    v(-k)*v(-p)*(-exp(-1j*p[0])*1j*v(k+p) + exp(1j*(k[0]+p[0]))*u(k+p)),
                ],
            ], [
                [
                    u(-k)*u(-p)*(exp(-1j*p[0])*u(k+p) + exp(1j*(k[0]+p[0]))*1j*v(k+p)),
                    u(-k)*v(-p)*(-exp(-1j*p[0])*1j*u(k+p) + exp(1j*(k[0]+p[0]))*v(k+p)),
                ], [
                    u(-k)*u(-p)*(-exp(-1j*p[0])*1j*v(k+p) + exp(1j*(k[0]+p[0]))*u(k+p)),
                    u(-k)*v(-p)*(-exp(-1j*p[0])*v(k+p) - exp(1j*(k[0]+p[0]))*1j*u(k+p))
                ],
            ],
        ] for p in momenta_BZ],
        # -k, -p, k-p
        [[
            [
                [
                    v(-k)*u(k+p)*(exp(1j*(k[0]+p[0]))*1j*u(-p) - exp(-1j*p[0])*v(-p)),
                    v(-k)*v(k+p)*(exp(1j*(k[0]+p[0]))*u(-p) + exp(-1j*p[0])*1j*v(-p))
                ], [
                    v(-k)*u(k+p)*(exp(1j*(k[0]+p[0]))*v(-p) + exp(-1j*p[0])*1j*u(-p)),
                    v(-k)*v(k+p)*(-exp(1j*(k[0]+p[0]))*1j*v(-p) + exp(-1j*p[0])*u(-p)),
                ],
            ], [
                [
                    u(-k)*u(k+p)*(exp(1j*(k[0]+p[0]))*u(-p) + exp(-1j*p[0])*1j*v(-p)),
                    u(-k)*v(k+p)*(-exp(1j*(k[0]+p[0]))*1j*u(-p) + exp(-1j*p[0])*v(-p)),
                ], [
                    u(-k)*u(k+p)*(-exp(1j*(k[0]+p[0]))*1j*v(-p) + exp(-1j*p[0])*u(-p)),
                    u(-k)*v(k+p)*(-exp(1j*(k[0]+p[0]))*v(-p) - exp(-1j*p[0])*1j*u(-p))
                ],
            ],
        ] for p in momenta_BZ],
    ])
    assert np.allclose(
        cubic_verts_eigenspace_for_loop_momentum_minusk,
        expected_cubic_verts_eigenspace_for_loop_momentum_minusk)
    symmetrized_vertices_caa_minusk = \
        self_energies_old.__get_all_Wick_contractions_of_cubic_vertices_for_loop_momentum(
            cubic_verts_eigenspace_for_loop_momentum_minusk,
            np.array([[0, 0, 1], [0, 1, 1]]), (num_ks,))
    expected_symmetrized_vertices_caa_minusk = np.array([
        cubic_verts_eigenspace_for_loop_momentum_k[0, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace_for_loop_momentum_k[1, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[2, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace_for_loop_momentum_k[3, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[4, p_idx, 0, 1, 1] +
        cubic_verts_eigenspace_for_loop_momentum_k[5, p_idx, 0, 1, 1] \
        for p_idx, p in enumerate(momenta_BZ)
    ]).conj()
    assert np.allclose(
        symmetrized_vertices_caa_minusk[0],
        expected_symmetrized_vertices_caa_minusk.reshape(num_ks, 1, 1, 1))
        
    # check p,pp,p self-energies
    reg = 0.05
    self_energies_old = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, eigws, eigws_minus_k_minus_BZ,
        cubic_verts_eigenspace_for_loop_momentum_k,
        0, particle_hole_states, reg)
    expected_self_energies = np.array([
        0.5/num_ks * np.sum(np.array([
            np.abs(expected_symmetrized_vertices_cca_k[p_idx])**2 \
                / (freq - eigws[p_idx, 0] - eigws[(-k_idx-p_idx) % num_ks, 0] + 1j*reg) \
            for p_idx, p in enumerate(momenta_BZ)
        ], dtype=complex)) for freq in freqs
    ], dtype=complex).reshape(len(freqs), 1, 1)
    assert np.allclose(self_energies_old, expected_self_energies)
        



def test_honeycomb_FM_Heisenberg_with_DMI():
    latt = HoneycombLatticeA()
    D = np.array([0, 0, 0.1])
    inter = [
        NthNearestNeighborHeisenbergInteraction(latt, n=1, J=-1.0),
        DMInteraction(BravaisLattice.Edge(np.array([1, 0]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([0, -1]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([-1, 1]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([1, 0]), [1, 1]), D=-D),
        DMInteraction(BravaisLattice.Edge(np.array([0, -1]), [1, 1]), D=-D),
        DMInteraction(BravaisLattice.Edge(np.array([-1, 1]), [1, 1]), D=-D),
    ]
    # in-plane ground state polarization to get cubic vertices
    mod = models.Model(latt, inter, np.array([[1, 0, 0]]*2))

    # VERTICES REAL SPACE
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)

    # VERTICES MOMENTUM SPACE
    k = np.array([0.5, 0.5])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_for_loop_momentum = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ)
    
    # VERTICES EIGENSPACE
    _, eigvs_at_k = LSWT.get_eigensystem_momentum_space(mod, k)
    energies_BZ, eigvs_BZ = LSWT.get_eigensystem_for_Brillouin_zone(mod, N_BZ)
    _, eigvs_minus_k_minus_BZ = LSWT.get_eigensystem_for_loop_momentum(
        mod, -k, N_BZ)
    verts_eigenspace_for_loop_momentum = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ, verts_for_loop_momentum)
    
    freqs = np.linspace(0, 5, 11)
    energies_k_minus_BZ, _ = LSWT.get_eigensystem_for_loop_momentum(mod, k, N_BZ)
    reg = 0.05
    se_p_pp_p = compute_one_magnon_one_loop_self_energies_at_momentum(
        freqs, energies_BZ, energies_k_minus_BZ,
        verts_eigenspace_for_loop_momentum, 0, ["p", "pp", "p"], reg)
    
    pass