from magpy.lattice import *
from magpy import models
from magpy.interactions import *
from magpy.largeS import real_space, momentum_space, eigenspace, normal_order
from magpy.largeS import LSWT
from magpy.self_energies import bubble, tadpole, quartic_bubble, util
import numpy as np

from magpy.momenta_utils import MSQ, Momenta
from magpy.util import PAULI_MATRICES




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
    _, eigvs = LSWT.get_eigensystem_momentum_space(mod, np.zeros((1, 1)))
    eigvs = np.array([eigvs, eigvs, eigvs])
    verts_eigenspace = eigenspace.compute_magnon_Hamiltonian_with_permutations(
        eigvs, verts_mom_space)

    # NORMAL-ORDER AND SYMMETRIZE
    verts_eigenspace_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonian(verts_eigenspace)

    # SELF-ENERGY
    freqs = np.array([0.0]) # np.linspace(0, 5, 11)
    reg = 0.05
    expected_se_pp = np.array([
        0.5/(freq - 2. + 1j*reg) * np.array([
            [9./8, -1.],
            [-1., 9./8],
        ]) for freq in freqs
    ])
    se_p_pp_p = bubble.compute_one_magnon_self_energy(
        freqs, np.array([1., 1., 1., 1.]), np.array([1., 1., 1., 1.]),
        verts_eigenspace_nosym, 0.0, ["p", "pp", "p"], reg)
    
    assert np.allclose(se_p_pp_p, expected_se_pp)






def test_honeycomb_FM_Heisenberg_with_DMI_on_slab():
    latt = HoneycombLatticeA()
    latt = rearrange_sublattices(latt, (1, 0))
    D = np.array([0.4, 0, 0.1])
    inter = [
        NthNearestNeighborHeisenbergInteraction(latt, n=1, J=-1.0),
        DMInteraction(BravaisLattice.Edge(np.array([1, 0]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([0, -1]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([-1, 1]), [0, 0]), D=D),
        DMInteraction(BravaisLattice.Edge(np.array([1, 0]), [1, 1]), D=-D),
        DMInteraction(BravaisLattice.Edge(np.array([0, -1]), [1, 1]), D=-D),
        DMInteraction(BravaisLattice.Edge(np.array([-1, 1]), [1, 1]), D=-D),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]*2))

    N_slab = 2
    mod = models.add_custom_open_bc(
        mod, (N_slab,), 
        slab_surface_coords=np.array([[1, 0]]), 
        slab_normal_coords=np.array([[0, 1]]))
    
    num_ks = 100
    hisym_points = ["G", "K", "M", "K'", "G'"]
    momentum_path = mod.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        hisym_points, num_ks, custom_hisym_points={
            "G": np.array([0.]),
            "K": np.array([1/3]),
            "M": np.array([1/2]),
            "K'": np.array([2/3]),
            "G'": np.array([1.]),
        })
    
    k_idx = 20
    k = momentum_path.ks[k_idx]

    # REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)
    verts_real_space_by_sites = group_interactions_by_site(verts_real_space, filter_zero=True)
    
    # MOMENTUM SPACE
    N_BZ = 10
    momenta_kpath = Momenta.of(momentum_path)
    momenta_k = Momenta(k)
    momenta_BZ = Momenta.of_BZ(mod.lattice, (N_BZ,))
    momenta_minus_k_minus_BZ = Momenta.of_BZ(mod.lattice, (N_BZ,), trans=lambda k_BZ: -k-k_BZ)
    verts_mom_space = momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
        mod, Momenta.join(momenta_BZ, momenta_k))

    a = mod.lattice.bravais_vecs[0]
    nnn = [None for _ in range(2*N_slab)]
    nnn[0] = ((+1, 0), (-1, +2)) # y = 1
    for y in range(2, 2*N_slab - 2, 2): # odd y
        nnn[y] = ((+1, 0), (-1, +2), (0, -2))
    nnn[2*N_slab - 2] = ((+1, 0), (0, -2)) # y = 2N_slab - 1
    nnn[1] = ((-1, 0), (0, +2)) # y = 2
    for y in range(3, 2*N_slab - 1, 2): # even y
        nnn[y] = ((-1, 0), (1, -2), (0, +2))
    nnn[2*N_slab - 1] = ((-1, 0), (1, -2)) # y = 2N_slab
    
    expected_verts_mom_space_minuskminusq_q_k_cca = np.zeros((N_BZ, *((2*N_slab,)*3)), dtype=complex)
    expected_verts_mom_space_q_minuskminusq_k_cca = np.zeros((N_BZ, *((2*N_slab,)*3)), dtype=complex)

    for nq, q in enumerate(momenta_BZ.k_arrays[0]):
        for y in range(2*N_slab):
            for dx, dy in nnn[y]:
                expected_verts_mom_space_minuskminusq_q_k_cca[nq, y, y+dy, y+dy] \
                    += np.exp(1j*(k+q).dot(a)*dx)
                expected_verts_mom_space_minuskminusq_q_k_cca[nq, y, y+dy, y] \
                    -= np.exp(1j*q.dot(a)*dx)
                expected_verts_mom_space_q_minuskminusq_k_cca[nq, y, y+dy, y+dy] \
                    += np.exp(-1j*q.dot(a)*dx)
                expected_verts_mom_space_q_minuskminusq_k_cca[nq, y, y+dy, y] \
                    -= np.exp(-1j*(k+q).dot(a)*dx)

    expected_verts_mom_space_minuskminusq_q_k_cca *= -1j * D[0]/np.sqrt(2)
    expected_verts_mom_space_q_minuskminusq_k_cca *= -1j * D[0]/np.sqrt(2)

    #assert np.allclose(verts_mom_space.raw_quantity[0, :, 1::2, 1::2, 0::2], expected_verts_mom_space_minuskminusq_q_k_cca)
    #assert np.allclose(verts_mom_space.raw_quantity[2, :, 1::2, 1::2, 0::2], expected_verts_mom_space_q_minuskminusq_k_cca)

    # EIGENSPACE
    eigws_k, eigvs_k = LSWT.get_eigensystems_momentum_space(mod, momenta_k)
    eigws_BZ, eigvs_BZ = LSWT.get_eigensystems_momentum_space(mod, momenta_BZ)
    eigws_minus_k_minus_BZ, eigvs_minus_k_minus_BZ = LSWT.get_eigensystems_momentum_space(mod, momenta_minus_k_minus_BZ)
    verts_eigenspace = eigenspace.compute_magnon_Hamiltonians_with_permutations(
        mod, MSQ.join(eigvs_minus_k_minus_BZ, eigvs_BZ, eigvs_k), verts_mom_space
    )

    # NORMAL-ORDER AND SYMMETRIZE
    verts_eigenspace_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(verts_eigenspace)
    
    expected_verts_eigenspace_nosym_aaa = np.zeros((N_BZ, 2*N_slab, 2*N_slab, 2*N_slab), dtype=np.complex128)
    assert np.allclose(verts_eigenspace_nosym.raw_quantity[0b000], expected_verts_eigenspace_nosym_aaa)

    expected_verts_eigenspace_nosym_cca = np.zeros((N_BZ, 2*N_slab, 2*N_slab, 2*N_slab), dtype=complex)
    for nq, (q, eigvs_minus_k_minus_q, eigvs_q) in enumerate(zip(
        momenta_BZ.k_arrays[0], eigvs_minus_k_minus_BZ.raw_quantity[0], eigvs_BZ.raw_quantity[0]
    )):
        for n in range(2*N_slab):
            for m in range(2*N_slab):
                for l in range(2*N_slab):
                    expected_verts_eigenspace_nosym_cca[nq, n, m, l] = sum(
                        sum(
                            + np.exp(1j*(k+q).dot(a)*dx) * eigvs_minus_k_minus_q[2*y+1,2*n+1] * eigvs_q[2*(y+dy)+1,2*m+1] * eigvs_k.raw_quantity[0][2*(y+dy),2*l] \
                            - np.exp(1j*q.dot(a)*dx) * eigvs_minus_k_minus_q[2*y+1,2*n+1] * eigvs_q[2*(y+dy)+1,2*m+1] * eigvs_k.raw_quantity[0][2*y,2*l] \
                            + np.exp(-1j*q.dot(a)*dx) * eigvs_minus_k_minus_q[2*(y+dy)+1,2*n+1] * eigvs_q[2*y+1,2*m+1] * eigvs_k.raw_quantity[0][2*(y+dy),2*l] \
                            - np.exp(-1j*(k+q).dot(a)*dx) * eigvs_minus_k_minus_q[2*(y+dy)+1,2*n+1] * eigvs_q[2*y+1,2*m+1] * eigvs_k.raw_quantity[0][2*y,2*l] \
                            for dx, dy in nnn[y]
                        ) for y in range(2*N_slab)
                    )
    expected_verts_eigenspace_nosym_cca *= -D[0] * 1j/2/np.sqrt(2)

    assert np.allclose(verts_eigenspace_nosym.raw_quantity[0b110], expected_verts_eigenspace_nosym_cca)

    freqs = np.linspace(0.0, 6.0, 6)
    T, reg = 0, 0.01
    se_p_pp_p = bubble.compute_one_magnon_self_energy(
        freqs, eigws_BZ.raw_quantity[0], eigws_minus_k_minus_BZ.raw_quantity[0],
        verts_eigenspace_nosym.raw_quantity, T, ["p", "pp", "p"], reg)
    
    expected_se_p_pp_p = np.zeros((len(freqs), 2*N_slab, 2*N_slab), dtype=complex)
    for nfreq, freq in enumerate(freqs):
        for nq, (q, eigws_minus_k_minus_q, eigws_q) in enumerate(zip(
            momenta_BZ.k_arrays[0], eigws_minus_k_minus_BZ.raw_quantity[0], eigws_BZ.raw_quantity[0]
        )):
            for l in range(2*N_slab):
                for l_ in range(2*N_slab):
                    for n in range(2*N_slab):
                        for m in range(2*N_slab):
                            expected_se_p_pp_p[nfreq, n, m] += \
                                expected_verts_eigenspace_nosym_cca[nq,l_,l,n] \
                                * expected_verts_eigenspace_nosym_cca[nq,l_,l,m].conj() \
                                / (freq - eigws_q[2*l] - eigws_minus_k_minus_q[2*l_] + 1j*reg)
    expected_se_p_pp_p *= 2 / N_BZ

    assert np.allclose(se_p_pp_p, expected_se_p_pp_p)




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
    expected_se_hh = -np.array(1/16 * B_xy_sq / (freqs + 2*B[2] + 1j*reg)) \
        .reshape((len(freqs), 1, 1))
    
    se_p_pp_p = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "pp", "p"], reg)
    assert np.allclose(se_p_pp_p, expected_se_pp)
    
    se_p_ph_p = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "ph", "p"], reg)
    assert np.allclose(se_p_ph_p, np.zeros(len(freqs)))
    
    se_p_hh_p = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "hh", "p"], reg)
    assert np.allclose(se_p_hh_p, np.zeros(len(freqs)))
    
    se_p_pp_h = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "pp", "h"], reg)
    assert np.allclose(se_p_pp_h, np.zeros(len(freqs)))
    
    se_p_ph_h = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "ph", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_p_ph_h = bubble.compute_one_magnon_self_energy(
        freqs, energies_BZ, energies_minus_k_minus_BZ,
        verts_eigenspace_nosym.raw_quantity, T, ["p", "hh", "h"], reg)
    assert np.allclose(se_p_ph_h, np.zeros(len(freqs)))
    
    se_h_hh_h = bubble.compute_one_magnon_self_energy(
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
    sigma_x, sigma_y, sigma_z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    h0, Delta = 3.0, lambda k: -np.cos(k[0])
    H_LSWT_along_kpath = LSWT.compute_LSWT_Hamiltonians_momentum_space_BdG(mod, Momenta.of(kpath), strip=True).raw_quantity
    expected_H_LSWT_along_kpath = np.array([
        h0*np.eye(2) + Delta(k)*sigma_y \
        for k in kpath.ks
    ], dtype=complex)
    assert np.allclose(H_LSWT_along_kpath, expected_H_LSWT_along_kpath)
    
    eigws, eigvs = LSWT.get_eigensystems_momentum_space(mod, Momenta.of(kpath), strip=True)
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
    eigvs_cubic_vert = [eigvs_minus_k_minus_BZ, eigvs, eigvs[k_idx]]
    cubic_verts_eigenspace = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(
            mod, 
            MSQ(
                eigvs_cubic_vert, 
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
        
    # check p,pp,p bubble self-energies
    freqs = np.linspace(0, 5, 5, endpoint=False)
    particle_hole_states = [["p"], ["p", "p"], ["p"]]
    
    reg = 0.05
    eigws_se, eigvs_se = LSWT.get_eigensystems_momentum_space(mod, Momenta(-k-momenta_BZ, momenta_BZ))
    eigws_minus_k_minus_BZ = eigws_se.raw_quantity[0]
    eigws_BZ = eigws_se.raw_quantity[1]
    eigvs_BZ = eigvs_se.raw_quantity[1]
    self_energies_bubble_pp = bubble.compute_one_magnon_self_energy(
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
    assert np.allclose(self_energies_bubble_pp, expected_self_energies)

    # check tadpole self-energies
    eigw_Gamma, eigv_Gamma = eigws_BZ[0], eigvs_BZ[0]
    cubic_verts_k0k = cubic_verts_eigenspace_nosym.raw_quantity[:, 0]
    linear_comm_terms = normal_order.compute_commutator_terms_with_permutations(
        mod, [], [np.array([eigv_Gamma])], momenta_BZ, eigvs_BZ,
        interaction_Hamiltonian_real_space=cubic_verts_real_space)
    linear_comm_terms_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(linear_comm_terms.reshape((1, 1, 2)))[:, 0]
    self_energies_tadpole_p = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigvs[k_idx], cubic_verts_k0k,
        [eigvs_cubic_vert[0][0], eigvs_cubic_vert[1][0], eigvs_cubic_vert[2]],
        linear_comm_terms_nosym, eigv_Gamma,
        (len(momenta_BZ),), 0.0, ["p", "p", "p"], reg=0.05)
    self_energies_tadpole_h = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigvs[k_idx], cubic_verts_k0k, 
        [eigvs_cubic_vert[0][0], eigvs_cubic_vert[1][0], eigvs_cubic_vert[2]],
        linear_comm_terms_nosym, eigv_Gamma,
        (len(momenta_BZ),), 0.0, ["p", "h", "p"], reg=0.05)
    
    gauge_trafo_comm_terms = np.array([1j, 1])
    gauge_trafo_in = np.array([-1, np.exp(1j*np.pi/4)])
    gauge_trafo_internal = np.array([np.exp(2j/3*np.pi), -1j])
    gauge_trafo_out = np.array([1, np.exp(1j/5*np.pi)])
    linear_comm_terms_different_gauge = normal_order.compute_commutator_terms_with_permutations(
        mod, [], [np.array([gauge_trafo_comm_terms*eigv_Gamma])], momenta_BZ, eigvs_BZ,
        interaction_Hamiltonian_real_space=cubic_verts_real_space)
    linear_comm_terms_different_gauge_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(linear_comm_terms_different_gauge.reshape((1, 1, 2)))[:, 0]
    self_energies_tadpole_p_different_gauge = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigvs[k_idx],
        np.einsum(
            "i,i...->i...", 
            np.kron(gauge_trafo_out, np.kron(gauge_trafo_internal, gauge_trafo_in)), cubic_verts_k0k
        ), [
            gauge_trafo_out*eigvs_cubic_vert[0][0], 
            gauge_trafo_internal*eigvs_cubic_vert[1][0], 
            gauge_trafo_in*eigvs_cubic_vert[2]
        ],
        linear_comm_terms_different_gauge_nosym, gauge_trafo_comm_terms*eigv_Gamma,
        (len(momenta_BZ),), 0.0, ["p", "p", "p"], reg=0.05)
    self_energies_tadpole_h_different_gauge = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigvs[k_idx],
        np.einsum(
            "i,i...->i...", 
            np.kron(gauge_trafo_out, np.kron(gauge_trafo_internal, gauge_trafo_in)), cubic_verts_k0k),
        [
            gauge_trafo_out*eigvs_cubic_vert[0][0], 
            gauge_trafo_internal*eigvs_cubic_vert[1][0], 
            gauge_trafo_in*eigvs_cubic_vert[2]
        ],
        linear_comm_terms_different_gauge_nosym, gauge_trafo_comm_terms*eigv_Gamma,
        (len(momenta_BZ),), 0.0, ["p", "h", "p"], reg=0.05)
    
    assert np.allclose(self_energies_tadpole_p, self_energies_tadpole_p_different_gauge)
    assert np.allclose(self_energies_tadpole_h, self_energies_tadpole_h_different_gauge)

        



def test_tadpole_gauge_invariance():
    np.random.seed(2)
    freqs = np.linspace(0, 1, 120)
    eigw_Gamma = np.array([1., -1., 1., -1.])
    eigv_Gamma = np.identity(4)
    eigv_k = np.identity(4)
    eigv_minus_k = np.identity(4)
    cubic_verts_k0k = np.zeros((8, 2, 2, 2), dtype=np.complex128)
    cubic_verts_k0k[0b110] = np.ones((2, 2, 2))
    eigvs_cubic_vert = np.array([eigv_minus_k, eigv_Gamma, eigv_k])
    linear_comm_terms = np.ones((2, 2), dtype=np.complex128)
    eigv_comm_term = eigv_Gamma

    self_energies_tadpole_p = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigv_k, cubic_verts_k0k, eigvs_cubic_vert,
        linear_comm_terms, eigv_comm_term,
        (1, 1, 1), 0, ["p", "p", "p"], reg=0.05,
    )

    # gauge_trafo_comm_terms = np.diag([1j, 1, 1, 1])
    # gauge_trafo_in = np.diag([-1, np.exp(1j*np.pi/4), 1, 1])
    # gauge_trafo_internal = np.diag([np.exp(2j/3*np.pi), -1j, 1, 1])
    # gauge_trafo_out = np.diag([1, np.exp(1j/5*np.pi), 1, 1])
    gauge_trafo_comm_terms = np.zeros((4, 4), dtype=np.complex128)
    gauge_trafo_comm_terms[::2,::2] = np.array([
        [np.cos(np.pi/4), -1j*np.sin(np.pi/4)],
        [np.sin(np.pi/4), 1j*np.cos(np.pi/4)],
    ])
    gauge_trafo_comm_terms[1::2,1::2] = np.eye(2)
    gauge_trafo_in = np.zeros((4, 4), dtype=np.complex128)
    gauge_trafo_in[::2,::2] = np.array([
        [np.cos(np.pi/4), -1j*np.sin(np.pi/4)],
        [np.sin(np.pi/4), 1j*np.cos(np.pi/4)],
    ])
    gauge_trafo_in[1::2,1::2] = np.array([
        [np.cos(np.pi/3), -np.sin(np.pi/3)],
        [np.sin(np.pi/3), np.cos(np.pi/3)],
    ])
    gauge_trafo_internal = np.diag([1, 1, 1, 1])
    gauge_trafo_out = np.zeros((4, 4), dtype=np.complex128)
    gauge_trafo_out[::2,::2] = np.array([
        [np.cos(np.pi/4), 1j*np.sin(np.pi/4)],
        [1j*np.sin(np.pi/4), np.cos(np.pi/4)],
    ])
    gauge_trafo_out[1::2,1::2] = np.array([
        [-1j*np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4), 1j*np.cos(np.pi/4)],
    ])
    cubic_verts_k0k_different_gauge = np.zeros((8, 2, 2, 2), dtype=np.complex128)
    for n in range(8):
        cubic_verts_k0k_different_gauge[n] = np.einsum(
            "IJK,Ii,Jj,Kk",
            cubic_verts_k0k[n],
            gauge_trafo_out[(n&0b100)>>2::2,(n&0b100)>>2::2], 
            gauge_trafo_internal[(n&0b010)>>1::2,(n&0b010)>>1::2],
            gauge_trafo_in[(n&0b001)>>0::2,(n&0b001)>>0::2],
        )
    linear_comm_terms_different_gauge = np.zeros((2, 2), dtype=np.complex128)
    for n in range(2):
        linear_comm_terms_different_gauge[n] = np.einsum(
            "I,Ii",
            linear_comm_terms[n],
            gauge_trafo_comm_terms[n::2, n::2],
        )

    self_energies_tadpole_p_different_gauge = tadpole.compute_one_magnon_self_energy(
        freqs, eigw_Gamma, eigv_k, 
        cubic_verts_k0k_different_gauge, [
            eigvs_cubic_vert[0] @ gauge_trafo_out, 
            eigvs_cubic_vert[1] @ gauge_trafo_internal, 
            eigvs_cubic_vert[2] @ gauge_trafo_in,
        ],
        linear_comm_terms_different_gauge, 
        eigv_Gamma @ gauge_trafo_comm_terms,
        (1, 1, 1), 0, ["p", "p", "p"], reg=0.05,
    )
    
    assert np.allclose(self_energies_tadpole_p, self_energies_tadpole_p_different_gauge)




def test_tadpole_gauge_invariance_with_degeneracies():
    J = -1.0
    Bz = 1.0
    Bx1, Bx2 = 0.7, 0.5

    ladder_lattice = ChainLattice(2, edges=[
        BravaisLattice.Edge(np.array([1]), np.array([0, 0])),   # lower ladder rail
        BravaisLattice.Edge(np.array([1]), np.array([1, 1])),   # upper ladder rail
        BravaisLattice.Edge(np.array([0]), np.array([0, 1])),   # ladder rung
    ])
    model = models.Model(ladder_lattice, interactions=[
        # Interaction([
        #     BravaisLattice.Site(np.array([0]), 0),
        #     BravaisLattice.Site(np.array([1]), 0),
        # ], interaction_tensor=J*np.eye(3)),
        # Interaction([
        #     BravaisLattice.Site(np.array([0]), 1),
        #     BravaisLattice.Site(np.array([1]), 1),
        # ], interaction_tensor=J*np.eye(3)),
        MagneticField(ladder_lattice, sublattice_index=0, B=np.array([Bx1, 0, Bz])),
        MagneticField(ladder_lattice, sublattice_index=1, B=np.array([Bx2, 0, Bz])),
    ], classical_ground_state=np.array([[0, 0, 1], [0, 0, 1]]))

    kpath = ladder_lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "Gamma'"], 1, {
            "Gamma": np.array([0]),
            "Gamma'": np.array([1]),
        })
    N_BZ = (1,)
    momenta_kpath = Momenta.of(kpath)
    momenta_BZ = Momenta.of_BZ(ladder_lattice, N_BZ)
    momenta_kpath_minus_BZ = Momenta(np.array([
        -k - ladder_lattice.reciprocal_lattice.sample_inverse_unit_cell(N_BZ, as_meshgrid=False) \
        for k in kpath.ks
    ]))
    momenta_Gamma = Momenta(np.zeros(1))

    eigws_kpath, eigvs_kpath = LSWT.get_eigensystems_momentum_space(
        model, momenta_kpath)
    eigws_BZ, eigvs_BZ = LSWT.get_eigensystems_momentum_space(
        model, momenta_BZ)
    eigws_minus_kpath_minus_BZ, eigvs_minus_kpath_minus_BZ = LSWT.get_eigensystems_momentum_space(
        model, momenta_kpath_minus_BZ)
    eigws_Gamma, eigvs_Gamma = LSWT.get_eigensystems_momentum_space(
        model, momenta_Gamma)
    
    def unitary_2x2_matrix(chi, psi, cos_theta, phi):
        sin_theta = np.sqrt(1 - cos_theta**2)
        return np.exp(1j*chi) * (
            np.cos(psi) * np.eye(2) +
            np.sin(psi) * 1j * np.einsum(
                "i,ijk->jk",
                np.array([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta]),
                PAULI_MATRICES,
            )
        )

    def random_regauge(eigvs):
        num_eigvs = int(np.prod(eigvs.shape[:-2]))
        chis = 2*np.pi*np.random.rand(num_eigvs, 2)
        psis = 2*np.pi*np.random.rand(num_eigvs, 2)
        cos_thetas = 2*np.random.rand(num_eigvs, 2) - 1
        phis = 2*np.pi*np.random.rand(num_eigvs, 2)

        eigvs_flat = eigvs.reshape((num_eigvs, *eigvs.shape[-2:]))
        for n, (chi, psi, cos_theta, phi) in enumerate(zip(chis, psis, cos_thetas, phis)):
            hole_gauge = unitary_2x2_matrix(chi[0], psi[0], cos_theta[0], phi[0])
            particle_gauge = unitary_2x2_matrix(chi[1], psi[1], cos_theta[1], phi[1])
            eigvs_flat[n, ::2, ::2] = eigvs_flat[n, ::2, ::2] @ hole_gauge
            eigvs_flat[n, 1::2, 1::2] = eigvs_flat[n, 1::2, 1::2] @ particle_gauge

        return eigvs_flat.reshape(eigvs.shape)

    eigvs_Gamma.raw_quantity[0] = random_regauge(eigvs_Gamma.raw_quantity[0])
    eigvs_BZ.raw_quantity[0] = random_regauge(eigvs_BZ.raw_quantity[0])
    eigvs_minus_kpath_minus_BZ.raw_quantity[0] = random_regauge(eigvs_minus_kpath_minus_BZ.raw_quantity[0])
    
    linear_vert_mom_space = momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
        model, Momenta())
    linear_vert_eigenspace = eigenspace.compute_magnon_Hamiltonians_with_permutations(
        model, eigvs_Gamma, linear_vert_mom_space)
    linear_vert_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
        linear_vert_eigenspace)
    
    cubic_vert_mom_space = momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
        model, Momenta.join(momenta_BZ, momenta_kpath))
    cubic_vert_eigenspace = eigenspace.compute_magnon_Hamiltonians_with_permutations(
        model, MSQ.join(eigvs_minus_kpath_minus_BZ, eigvs_BZ, eigvs_kpath), cubic_vert_mom_space)
    cubic_vert_nosym = normal_order.normal_order_and_symmetrize_magnon_Hamiltonians(
        cubic_vert_eigenspace)

    freqs = np.array([0.])
    expected_se_tadpoles_p = 1/np.prod(N_BZ) * np.array([
        [np.diag([Bx1, Bx2])**2 / (4*Bz)] for _ in range(len(kpath.ks))
    ])
    expected_se_tadpoles_h = 1/np.prod(N_BZ) * np.array([
        [np.diag([Bx1, Bx2])**2 / (4*Bz)] for _ in range(len(kpath.ks))
    ])

    se_tadpoles_p = np.zeros((len(kpath.ks), 1, 2, 2))
    se_tadpoles_h = np.zeros((len(kpath.ks), 1, 2, 2))
    for k_idx, (k, eigvs_k, eigvs_k_minus_BZ, cubic_vert_k0k) in enumerate(zip(
        kpath.ks, 
        eigvs_kpath.raw_quantity[0], 
        eigvs_minus_kpath_minus_BZ.raw_quantity[0], 
        np.swapaxes(cubic_vert_nosym.raw_quantity[:, 0], 0, 1),
    )):
        se_tadpoles_p[k_idx] = tadpole.compute_one_magnon_self_energy(
            freqs, eigws_Gamma.raw_quantity[0], eigvs_k,
            cubic_vert_k0k, [eigvs_k_minus_BZ[0], eigvs_BZ.raw_quantity[0][0], eigvs_k],
            linear_vert_nosym.raw_quantity, eigvs_Gamma.raw_quantity[0],
            int(np.prod(N_BZ)), T=0, ph_labels=["p", "p", "p"], reg=0.0,
        )
        se_tadpoles_h[k_idx] = tadpole.compute_one_magnon_self_energy(
            freqs, eigws_Gamma.raw_quantity[0], eigvs_k,
            cubic_vert_k0k, [eigvs_k_minus_BZ[0], eigvs_BZ.raw_quantity[0][0], eigvs_k],
            linear_vert_nosym.raw_quantity, eigvs_Gamma.raw_quantity[0],
            int(np.prod(N_BZ)), T=0, ph_labels=["p", "h", "p"], reg=0.0,
        )
    
    assert np.allclose(se_tadpoles_p, expected_se_tadpoles_p)
    assert np.allclose(se_tadpoles_h, expected_se_tadpoles_h)





def test_quartic_bubble():
    freqs = np.linspace(0, 1, 120)
    quadratic_commutator_terms = np.zeros((4, 1, 1))
    num_ks_BZ = 10

    quartic_bubble.compute_one_magnon_self_energy(
        freqs,
        np.identity(2),
        quadratic_commutator_terms,
        np.full((2, 2, 2), np.identity(2)),
        num_ks_BZ,
        T=0.0, 
        ph_labels=["p", "p"],
    )



def test_diagram_signs():
    # bubbles
    assert util.compute_diagram_sign(order=2, num_internal_propagators=2) == -1
    # stubs
    assert util.compute_diagram_sign(order=2, num_internal_propagators=1) == 1
    # quadratic insertion
    assert util.compute_diagram_sign(order=1, num_internal_propagators=0) == 1



def test_num_Wick_contractions():
    PARTICLE, HOLE = 1, 0

    # particle-particle bubble
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, HOLE, HOLE], [PARTICLE, PARTICLE, HOLE]],
        ph_idxs_loops=[[PARTICLE, PARTICLE]],
    ) == 2
    # particle-hole bubble
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, HOLE, PARTICLE], [PARTICLE, HOLE, HOLE]],
        ph_idxs_loops=[[PARTICLE, HOLE]],
    ) == 4
    # hole-hole bubble
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, PARTICLE, PARTICLE], [HOLE, HOLE, HOLE]],
        ph_idxs_loops=[[HOLE, HOLE]],
    ) == 18

    # particle stub
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, HOLE, HOLE]],
        ph_idxs_loops=[],
    ) == 2
    # hole stub
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, PARTICLE, HOLE]],
        ph_idxs_loops=[],
    ) == 2

    # quadratic insertion
    assert util.compute_num_Wick_contractions(
        ph_idxs_verts=[[PARTICLE, HOLE]],
        ph_idxs_loops=[],
    ) == 1