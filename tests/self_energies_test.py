from magpy.lattice import *
from magpy import models
from magpy.interactions import *
from magpy.largeS import real_space, momentum_space, eigenspace, normal_order
from magpy.largeS import LSWT
from magpy import self_energies
import numpy as np

from magpy.momenta_utils import MSQ, Momenta




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
    verts_eigenspace_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonian(verts_eigenspace)

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
        verts_eigenspace_nosym, 0.0, ["p", "pp", "p"], reg)
    
    assert np.allclose(se_p_pp_p, expected_se_pp)


def test_field_orthogonal_to_quantization_direction():
    latt = SquareLattice()
    B = np.array([1, 2, 3])
    inter = [
        MagneticField(latt, 0, B),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))

    # VERTICES REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)

    # VERTICES MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice \
        .sample_inverse_unit_cell(N_BZ) \
        .transpose((1, 2, 0))
    momenta = Momenta(momenta_BZ, k)
    verts_mom_space = momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
        mod, momenta, verts_real_space)
    
    # VERTICES EIGENSPACE
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, Momenta(-k-momenta_BZ, momenta_BZ, k))
    verts_eigenspace = eigenspace.compute_magnon_Hamiltonians_with_permutations(mod, eigvs, verts_mom_space)
    verts_eigenspace_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(verts_eigenspace)
    
    # SELF-ENERGY
    # TODO FIX ENERGIES OF LOOP MOMENTA!!!
    energies_minus_k_minus_BZ = eigws.raw_quantity[0]
    energies_BZ = eigws.raw_quantity[1]
    freqs = np.linspace(0, 5, 11)
    reg = 0.05
    T = 0.0
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
    
    se_p_pp_p = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "pp", "p"], reg)
    assert np.allclose(se_p_pp_p, expected_se_pp)
    
    se_p_ph_p = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "ph", "p"], reg)
    assert np.allclose(se_p_ph_p, np.zeros(len(freqs)))
    
    se_p_hh_p = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "hh", "p"], reg)
    assert np.allclose(se_p_hh_p, np.zeros(len(freqs)))
    
    se_p_pp_h = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "pp", "h"], reg)
    assert np.allclose(se_p_pp_h, np.zeros(len(freqs)))
    
    se_p_ph_h = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "ph", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_p_ph_h = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "hh", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_h_hh_h = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["h", "hh", "h"], reg)
    assert np.allclose(se_h_hh_h, expected_se_hh)



def test_1D_BdG_chain_with_cubic_interaction():
    latt = ChainLattice(1)
    mod = models.Model(latt, [
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

    # LSWT
    H_LSWT_real_space = real_space.compute_magnon_Hamiltonian(mod, order=2)
    assert np.allclose(
        H_LSWT_real_space[0].interaction_tensor,
        -0.5j*np.array([[1, 0], [0, 0]]))
    assert np.allclose(
        H_LSWT_real_space[1].interaction_tensor,
        0.5j*np.array([[0, 0], [0, 1]]))
    assert np.allclose(
        H_LSWT_real_space[2].interaction_tensor,
        np.array([[0, 0], [3.0, 0]]))
    
    num_ks = 10
    kpath = ReciprocalLattice.MomentumPath(
        np.linspace(0, 2*np.pi, num_ks, endpoint=False).reshape(num_ks, 1))
    sigma_y, sigma_z = np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    h0, Delta = 3.0, lambda k: -np.cos(k[0])
    H_LSWT_along_kpath = LSWT.compute_LSWT_Hamiltonians_momentum_space_BdG(mod, Momenta.of(kpath)).raw_quantity
    expected_H_LSWT_along_kpath = np.array([
        h0*np.eye(2) + Delta(k)*sigma_y \
        for k in kpath.ks
    ], dtype=complex)
    assert np.allclose(H_LSWT_along_kpath, expected_H_LSWT_along_kpath)
    
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, Momenta.of(kpath))
    eigws, eigvs = eigws.raw_quantity, eigvs.raw_quantity
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

    # INTERACTIONS
    #
    # make up some real space interaction vertices
    # a_i^† a_i a_j + a_i^† a_j^† a_i
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
    momenta = Momenta(momenta_BZ, k)
    cubic_verts_mom_space = \
        momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(mod, momenta, cubic_verts_real_space)
    caa = np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [0, 0]],
    ])
    cca = np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ])
    exp = np.exp
    expected_cubic_verts_mom_space = np.array([
        [exp(1j*k[0])*caa + exp(1j*p[0])*cca for p in momenta_BZ],
        [exp(1j*p[0])*caa + exp(1j*k[0])*cca for p in momenta_BZ],
        [exp(1j*k[0])*caa + exp(-1j*(k[0]+p[0]))*cca for p in momenta_BZ],
        [exp(-1j*(k[0]+p[0]))*caa + exp(1j*k[0])*cca for p in momenta_BZ],
        [exp(1j*p[0])*caa + exp(-1j*(k[0]+p[0]))*cca for p in momenta_BZ],
        [exp(-1j*(k[0]+p[0]))*caa + exp(1j*p[0])*cca for p in momenta_BZ],
    ])
    assert np.allclose(
        cubic_verts_mom_space.raw_quantity,
        expected_cubic_verts_mom_space)
    
    # check LSWT eigenspace interaction vertices
    eigvs_minus_k_minus_BZ = np.array([
        eigvs[(-k_idx-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    cubic_verts_eigenspace = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(
            mod, 
            MSQ(
                [eigvs_minus_k_minus_BZ, eigvs, eigvs[k_idx]], 
                Momenta(-k-momenta_BZ, momenta_BZ, k)
            ),
            cubic_verts_mom_space
        )

    expected_cubic_verts_eigenspace = np.array([
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
        cubic_verts_eigenspace.raw_quantity,
        expected_cubic_verts_eigenspace)

    # check symmetrized and normal-ordered eigenspace interaction vertices
    cubic_verts_eigenspace_nosym = \
        normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
            cubic_verts_eigenspace)
    expected_cubic_verts_eigenspace_nosym = np.array([
        [[
            (expected_cubic_verts_eigenspace[0, np, 0, 0, 0] + \
            expected_cubic_verts_eigenspace[1, np, 0, 0, 0] + \
            expected_cubic_verts_eigenspace[2, np, 0, 0, 0] + \
            expected_cubic_verts_eigenspace[3, np, 0, 0, 0] + \
            expected_cubic_verts_eigenspace[4, np, 0, 0, 0] + \
            expected_cubic_verts_eigenspace[5, np, 0, 0, 0]) / 6
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 0, 0, 1] + \
            expected_cubic_verts_eigenspace[1, np, 0, 1, 0] + \
            expected_cubic_verts_eigenspace[2, np, 0, 0, 1] + \
            expected_cubic_verts_eigenspace[3, np, 0, 1, 0] + \
            expected_cubic_verts_eigenspace[4, np, 1, 0, 0] + \
            expected_cubic_verts_eigenspace[5, np, 1, 0, 0]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 0, 1, 0] + \
            expected_cubic_verts_eigenspace[1, np, 0, 0, 1] + \
            expected_cubic_verts_eigenspace[2, np, 1, 0, 0] + \
            expected_cubic_verts_eigenspace[3, np, 1, 0, 0] + \
            expected_cubic_verts_eigenspace[4, np, 0, 0, 1] + \
            expected_cubic_verts_eigenspace[5, np, 0, 1, 0]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 0, 1, 1] + \
            expected_cubic_verts_eigenspace[1, np, 0, 1, 1] + \
            expected_cubic_verts_eigenspace[2, np, 1, 0, 1] + \
            expected_cubic_verts_eigenspace[3, np, 1, 1, 0] + \
            expected_cubic_verts_eigenspace[4, np, 1, 0, 1] + \
            expected_cubic_verts_eigenspace[5, np, 1, 1, 0]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 1, 0, 0] + \
            expected_cubic_verts_eigenspace[1, np, 1, 0, 0] + \
            expected_cubic_verts_eigenspace[2, np, 0, 1, 0] + \
            expected_cubic_verts_eigenspace[3, np, 0, 0, 1] + \
            expected_cubic_verts_eigenspace[4, np, 0, 1, 0] + \
            expected_cubic_verts_eigenspace[5, np, 0, 0, 1]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 1, 0, 1] + \
            expected_cubic_verts_eigenspace[1, np, 1, 1, 0] + \
            expected_cubic_verts_eigenspace[2, np, 0, 1, 1] + \
            expected_cubic_verts_eigenspace[3, np, 0, 1, 1] + \
            expected_cubic_verts_eigenspace[4, np, 1, 1, 0] + \
            expected_cubic_verts_eigenspace[5, np, 1, 0, 1]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 1, 1, 0] + \
            expected_cubic_verts_eigenspace[1, np, 1, 0, 1] + \
            expected_cubic_verts_eigenspace[2, np, 1, 1, 0] + \
            expected_cubic_verts_eigenspace[3, np, 1, 0, 1] + \
            expected_cubic_verts_eigenspace[4, np, 0, 1, 1] + \
            expected_cubic_verts_eigenspace[5, np, 0, 1, 1]) / 2
        ] for np, p in enumerate(momenta_BZ)],
        [[
            (expected_cubic_verts_eigenspace[0, np, 1, 1, 1] + \
            expected_cubic_verts_eigenspace[1, np, 1, 1, 1] + \
            expected_cubic_verts_eigenspace[2, np, 1, 1, 1] + \
            expected_cubic_verts_eigenspace[3, np, 1, 1, 1] + \
            expected_cubic_verts_eigenspace[4, np, 1, 1, 1] + \
            expected_cubic_verts_eigenspace[5, np, 1, 1, 1]) / 6
        ] for np, p in enumerate(momenta_BZ)],
    ]).reshape((8, 10, 1, 1, 1))
    assert np.allclose(cubic_verts_eigenspace_nosym.raw_quantity, expected_cubic_verts_eigenspace_nosym)
    
    # check all Wick contractions of the pp-bubble, encapsulated in the
    # creator-creator-annihilator (cca) symmetrized interaction vertex
    # eigws_minus_k_minus_BZ = np.array([
    #     eigws[(-k_idx-p_idx) % num_ks] \
    #     for p_idx in range(num_ks)
    # ])
    symmetrized_vertices_cca_k = cubic_verts_eigenspace_nosym.raw_quantity[0b110]
    expected_symmetrized_vertices_cca_k = 1/2 * np.array([
        v(k)*v(-k-p) * (-exp(-1j*(k[0]+p[0]))*1j*v(p) + exp(1j*p[0])*u(p)) + \
        v(k)*v(p) * (-exp(1j*p[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)) + \
        u(p)*v(-k-p) * (-exp(-1j*(k[0]+p[0]))*1j*u(k) + exp(1j*k[0])*v(k)) + \
        u(-k-p)*v(p) * (-exp(1j*p[0])*1j*u(k) + exp(1j*k[0])*v(k)) + \
        u(-k-p)*u(k) * (-exp(1j*k[0])*1j*v(p) + exp(1j*p[0])*u(p)) + \
        u(p)*u(k) * (-exp(1j*k[0])*1j*v(-k-p) + exp(-1j*(k[0]+p[0]))*u(-k-p)) \
        for p in momenta_BZ
    ])
    expected_symmetrized_vertices_cca_k_2 = 1/2 * np.array([
        cubic_verts_eigenspace.raw_quantity[0, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace.raw_quantity[1, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace.raw_quantity[2, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace.raw_quantity[3, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace.raw_quantity[4, p_idx, 0, 1, 1] +
        cubic_verts_eigenspace.raw_quantity[5, p_idx, 0, 1, 1] \
        for p_idx, p in enumerate(momenta_BZ)
    ])
    assert np.allclose(
        expected_symmetrized_vertices_cca_k,
        expected_symmetrized_vertices_cca_k_2)
    assert np.allclose(
        symmetrized_vertices_cca_k,
        expected_symmetrized_vertices_cca_k.reshape(num_ks, 1, 1, 1))
    # assert np.allclose(
    #     symmetrized_vertices_cca_k[1],
    #     symmetrized_vertices_cca_k[0].conj())
    # assert np.allclose(
    #     symmetrized_vertices_cca_k[0],
    #     expected_symmetrized_vertices_cca_k.reshape(num_ks, 1, 1, 1))

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
    swapped_momenta = Momenta(-momenta_BZ, k)
    cubic_verts_mom_space_swapped_k = \
        momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
            mod, swapped_momenta, cubic_verts_real_space)
    cubic_verts_eigenspace_swapped_k = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(
            mod, 
            MSQ(
                [eigvs_minus_k_plus_BZ, eigvs_minus_BZ, eigvs[k_idx]], 
                Momenta(-k+momenta_BZ, -momenta_BZ, k)
            ),
            cubic_verts_mom_space_swapped_k
        )
    cubic_verts_eigenspace_nosym_swapped_k = \
        normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
            cubic_verts_eigenspace_swapped_k)
    symmetrized_vertices_swapped_cca_k = \
        cubic_verts_eigenspace_nosym_swapped_k.raw_quantity[0b110]
    expected_symmetrized_vertices_swapped_cca_k = np.array([
        expected_symmetrized_vertices_cca_k[(-p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    assert np.allclose(
        symmetrized_vertices_swapped_cca_k,
        expected_symmetrized_vertices_swapped_cca_k.reshape(num_ks, 1, 1, 1))
    
    # # check if the annihilator-annihilator-creator (aac) symmetrized interaction
    # # vertex maps to the conjugate of the caa vertex with inverted momenta
    eigvs_plus_k_plus_BZ = np.array([
        eigvs[(k_idx+p_idx) % num_ks] \
        for p_idx in range(num_ks)
    ])
    momenta_minusk = Momenta(-momenta_BZ, -k)
    cubic_verts_mom_space_minusk = \
        momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
            mod, momenta_minusk, cubic_verts_real_space)
    cubic_verts_eigenspace_minusk = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(
            mod, 
            MSQ(
                [eigvs_plus_k_plus_BZ, eigvs_minus_BZ, eigvs[(-k_idx) % num_ks]], 
                Momenta(k+momenta_BZ, -momenta_BZ, -k)
            ),
            cubic_verts_mom_space_minusk
        )
    expected_cubic_verts_eigenspace_minusk = np.array([
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
        cubic_verts_eigenspace_minusk.raw_quantity,
        expected_cubic_verts_eigenspace_minusk)
    
    cubic_verts_eigenspace_nosym_minusk = \
        normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
            cubic_verts_eigenspace_minusk)
    symmetrized_vertices_aac_minusk = \
        cubic_verts_eigenspace_nosym_minusk.raw_quantity[0b001]
    expected_symmetrized_vertices_aac_minusk = 1/2 * np.array([
        cubic_verts_eigenspace.raw_quantity[0, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace.raw_quantity[1, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace.raw_quantity[2, p_idx, 1, 1, 0] +
        cubic_verts_eigenspace.raw_quantity[3, p_idx, 1, 0, 1] +
        cubic_verts_eigenspace.raw_quantity[4, p_idx, 0, 1, 1] +
        cubic_verts_eigenspace.raw_quantity[5, p_idx, 0, 1, 1] \
        for p_idx, p in enumerate(momenta_BZ)
    ]).conj()
    assert np.allclose(
        symmetrized_vertices_aac_minusk,
        expected_symmetrized_vertices_aac_minusk.reshape(num_ks, 1, 1, 1))
        
    # check p,pp,p self-energies
    freqs = np.linspace(0, 5, 5, endpoint=False)
    particle_hole_states = [["p"], ["p", "p"], ["p"]]
    
    reg = 0.05
    eigws_se, _ = LSWT.get_eigensystems_momentum_space(mod, Momenta(-k-momenta_BZ, momenta_BZ))
    eigws_minus_k_minus_BZ = eigws_se.raw_quantity[0]
    eigws_BZ = eigws_se.raw_quantity[1]
    self_energies_old = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, eigws_BZ, eigws_minus_k_minus_BZ,
        cubic_verts_eigenspace_nosym.raw_quantity,
        0, particle_hole_states, reg)
    expected_self_energies = np.array([
        2.0/num_ks * np.sum(np.array([
            np.abs(expected_cubic_verts_eigenspace_nosym[0b110, p_idx])**2 \
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
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)

    # VERTICES MOMENTUM SPACE
    k = np.array([0.5, 0.5])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice \
        .sample_inverse_unit_cell(N_BZ) \
        .transpose([1, 2, 0])
    momenta_q_k = Momenta(momenta_BZ, k)
    momenta_minuskminusq_q_k = Momenta(-k-momenta_BZ, momenta_BZ, k)
    verts_mom_space = momentum_space \
        .compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
            mod, momenta_q_k, verts_real_space)
    
    # VERTICES EIGENSPACE
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, momenta_minuskminusq_q_k)
    verts_eigenspace = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(
            mod, eigvs, verts_mom_space)
    verts_eigenspace_nosym = \
        normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
            verts_eigenspace)
    
    freqs = np.linspace(0, 5, 11)
    energies_minus_k_minus_BZ = eigws.raw_quantity[0]
    energies_BZ = eigws.raw_quantity[1]
    reg = 0.05
    se_p_pp_p = self_energies.compute_one_magnon_self_energy_bubble(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, 0, ["p", "pp", "p"], reg)
    
    pass