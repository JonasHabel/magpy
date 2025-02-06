from magpy.lattice import *
from magpy import models
from magpy.interactions import *
from magpy.interaction_vertices import real_space as real_space_old            # refactor this old dependency
from magpy.interaction_vertices import momentum_space as momentum_space_old    # refactor this old dependency
from magpy.interaction_vertices import eigenspace as eigenspace_old            # refactor this old dependency
from magpy.interaction_vertices.util import GET_CUBIC_PERMUTATIONS
from magpy import LSWT as LSWT_old # refactor this old dependency
from magpy.largeS import real_space, momentum_space, LSWT, eigenspace, normal_order
from magpy.correlators.magnon_correlators import compute_real_space_correlator_LSWT
import numpy as np
from magpy.momenta_utils import MSQ, Momenta
from magpy.util import permute


def test_two_site_quantum_dot_with_DMI():
    latt = DotLattice(2)
    mod = models.Model(latt, [
        DMInteraction(BravaisLattice.Edge(np.array([]), [0, 1]), D=np.array([1.0, 0, 0]))
    ], classical_ground_state=np.array([[0, 0, 1], [0, 0, 1]]))

    # REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)

    # expect 1/sqrt(2) [a_1 a_2^† a_2 - a_1^† a_1 a_2 + h.c.] 
    #      + 1/sqrt(8) [a_1^† a_1 a_1 - a_2^† a_2 a_2 + h.c.]
    expected_int_tensor_111 = 1/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [-1j, 0]],
    ])
    expected_int_tensor_222 = -1/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [-1j, 0]],
    ])
    expected_int_tensor_122 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [1j, 0]],
        [[0, 0], [0, 0]],
    ])
    expected_int_tensor_112 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[-1j, 0], [0, 0]],
    ])
    expected_int_tensor_221 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[0, -1j], [0, 0]],
    ])
    expected_int_tensor_211 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [1j, 0]],
    ])

    assert len(verts_real_space) == 6

    assert np.allclose(expected_int_tensor_111, verts_real_space[0].interaction_tensor)
    assert np.allclose(expected_int_tensor_222, verts_real_space[1].interaction_tensor)
    assert np.allclose(expected_int_tensor_122, verts_real_space[2].interaction_tensor)
    assert np.allclose(expected_int_tensor_112, verts_real_space[3].interaction_tensor)
    assert np.allclose(expected_int_tensor_221, verts_real_space[4].interaction_tensor)
    assert np.allclose(expected_int_tensor_211, verts_real_space[5].interaction_tensor)

    # MOMENTUM SPACE
    vert_mom_space = momentum_space.compute_magnon_Hamiltonian(
        mod, np.array([[0], [0], [0]]), verts_real_space)
    
    expected_vert_mom_space = 1/(2*np.sqrt(8)) * np.array([
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4j, 0]],
        [[1j, 0, -4j, 0], [-1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [4j, 0, 0, 0], [0, -4j, -1j, 0], [0, 0, 1j, 0]],
    ])

    assert np.allclose(vert_mom_space, expected_vert_mom_space)

    # LSWT EIGENSPACE
    _, eigvs = LSWT.get_eigensystem_momentum_space(mod, np.array([0]))
    eigvs = np.array([eigvs, eigvs, eigvs])
    
    vert_eigenspace = eigenspace.compute_magnon_Hamiltonian(
        eigvs, vert_mom_space)
    
    assert np.allclose(vert_eigenspace, expected_vert_mom_space)



def test_field_orthogonal_to_quantization_direction():
    latt = SquareLattice()
    B = np.array([1, 2, 3])
    inter = [
        MagneticField(latt, 0, B),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))

    # REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3)
    
    # expect -1/(2*sqrt(8)) ((Bx + iBy) a_i^† a_i^† a_i + h.c.)
    # up to gauge choice
    expected_int_tensor = -1/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[B[0] - 1j*B[1], 0], [B[0] + 1j*B[1], 0]],
    ])
    assert len(verts_real_space) == 1
    # the vertex connects sites i, i, i
    assert all(map(
        lambda site: np.all(site.bravais_coords == np.zeros(2)) \
                     and site.subl_idx == 0,
        verts_real_space[0].sites))
    int_tensor = verts_real_space[0].interaction_tensor
    # check hermiticity
    assert np.allclose(int_tensor[:, ::-1, :], np.conj(int_tensor))
    # check magnitude of matrix elements
    assert np.allclose(np.abs(int_tensor), np.abs(expected_int_tensor))

    # FIXED-KS MOMENTUM SPACE
    ks = np.array([[np.random.rand(), np.random.rand()],
                   [np.random.rand(), np.random.rand()]])
    ks = np.array([*ks, -ks[0]-ks[1]])
    vert_mom_space = momentum_space.compute_magnon_Hamiltonian(
        mod, ks, verts_real_space)
    # in this testcase, the real space and momentum space vertex coefficients
    # should be the same
    assert np.allclose(vert_mom_space, int_tensor)

    # FIXED-KS LSWT EIGENSPACE
    eigvs = [None]*len(ks)
    for n, k in enumerate(ks):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_magnon_Hamiltonian(
        eigvs, vert_mom_space
    )
    assert np.allclose(vert_eigenspace, np.einsum(
        "im,jn,kr,ijk", eigvs[0], eigvs[1], eigvs[2], vert_mom_space
    ))

    # BZ MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    num_ks_BZ = int(np.prod(N_BZ))
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_for_loop_momentum = \
        momentum_space_old.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ)
    
    assert np.allclose(verts_for_loop_momentum,
                       vert_mom_space[np.newaxis, np.newaxis, np.newaxis, ...])
    
    # BZ EIGENSPACE
    _, eigvs_at_k = LSWT.get_eigensystem_momentum_space(mod, k)
    _, eigvs_BZ = LSWT.get_eigensystems_momentum_space(
        mod, Momenta.of_BZ(mod.lattice, N_BZ))
    _, eigvs_minus_k_minus_BZ = LSWT.get_eigensystems_momentum_space(
        mod, Momenta.of_BZ(mod.lattice, N_BZ, trans=lambda q: -k-q))
    eigvs_BZ = eigvs_BZ.raw_quantity[0]
    eigvs_minus_k_minus_BZ = eigvs_minus_k_minus_BZ.raw_quantity[0]
    verts_eigenspace_for_loop_momentum = \
        eigenspace_old.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ, 
            verts_for_loop_momentum)
    
    for n, p in enumerate(GET_CUBIC_PERMUTATIONS()):
        assert np.allclose(verts_eigenspace_for_loop_momentum[n], np.einsum(
            "xyim,xyjn,xykr,xyijk->xymnr",
            *permute(np.array([
                eigvs_minus_k_minus_BZ, eigvs_BZ, 
                eigvs_at_k[np.newaxis, np.newaxis]
            ]), p),
            verts_for_loop_momentum[n]
        ))
    

    




def test_FM_Heisenberg():
    latt = HoneycombLatticeA()
    inter = [
        NthNearestNeighborHeisenbergInteraction(latt, 1, J=-1)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]*2))

    # REAL SPACE
    verts_real_space = real_space_old.compute_interaction_Hamiltonian(mod, order=3)
    # FM interaction does not yield any cubic terms
    assert all(map(
        lambda vert: np.all(vert.interaction_tensor == 0),
        verts_real_space))
    
    # FIXED-KS MOMENTUM SPACE
    ks = np.array([[np.random.rand(), np.random.rand()],
                   [np.random.rand(), np.random.rand()]])
    ks = np.array([*ks, -ks[0]-ks[1]])
    vert_mom_space = momentum_space_old.compute_interaction_Hamiltonian(
        mod, 3, ks, verts_real_space)
    assert np.all(vert_mom_space == 0)

    # FIXED-KS LSWT EIGENSPACE
    eigvs = [None]*len(ks)
    for n, k in enumerate(ks):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_magnon_Hamiltonian(
        eigvs, vert_mom_space
    )
    assert np.all(vert_eigenspace == 0)
    




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
    # in-plane polarization to get cubic vertices
    mod = models.Model(latt, inter, np.array([[1, 0, 0]]*2))


    # REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, order=3, output_compression=None)


    # FM interaction does not yield any cubic terms
    # DMI does yield cubic terms
    assert len(verts_real_space) == 6*6
    # first, the iii- and jjj-terms (each up to gauge choice)
    expected_int_tensor_iii = D[2] * 1j/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [-1, 0]],
    ])
    expected_int_tensor_jjj = -expected_int_tensor_iii
    for n in 6*np.arange(3):
        expected_subl_idx = 0
        for x in range(2):
            assert all(map(
                lambda site: site.subl_idx == expected_subl_idx,
                verts_real_space[n+x].sites))

        int_tensor_iii = verts_real_space[n].interaction_tensor
        int_tensor_jjj = verts_real_space[n+1].interaction_tensor
        # check hermiticity
        assert np.allclose(int_tensor_iii[:, ::-1, :], np.conj(int_tensor_iii))
        assert np.allclose(int_tensor_jjj[:, ::-1, :], np.conj(int_tensor_jjj))
        # check magnitude of matrix elements
        assert np.allclose(
            np.abs(int_tensor_iii), np.abs(expected_int_tensor_iii))
        assert np.allclose(
            np.abs(int_tensor_jjj), np.abs(expected_int_tensor_jjj))
        # check if they're the negative of the corresponding -D interactions
        assert np.allclose(
            int_tensor_iii, -verts_real_space[3*6 + n].interaction_tensor)
        assert np.allclose(
            int_tensor_jjj, -verts_real_space[3*6 + n+1].interaction_tensor)
        
    # next, the ijj-, iij-, jji- and jii-terms (each up to gauge choice)
    expected_int_tensor_ijj = D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [1, 0]],
        [[0, 0], [0, 0]],
    ])  # ai aj+ aj
    expected_int_tensor_iij = D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [0, 0]],
    ])  # ai+ ai aj
    expected_int_tensor_jji = -D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [0, 0]],
        [[0, 1], [0, 0]],
    ])  # aj+ aj ai+
    expected_int_tensor_jii = -D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ])  # aj+ ai+ ai
    for n in 6*np.arange(3):
        expected_subl_idx = 0
        for x in range(2, 6):
            assert all(map(
                lambda site: site.subl_idx == expected_subl_idx,
                verts_real_space[n+x].sites))
        int_tensor_ijj = verts_real_space[n+2].interaction_tensor
        int_tensor_iij = verts_real_space[n+3].interaction_tensor
        int_tensor_jji = verts_real_space[n+4].interaction_tensor
        int_tensor_jii = verts_real_space[n+5].interaction_tensor
        # check hermiticity
        int_tensor_jji_sym = int_tensor_jji + int_tensor_jji.transpose((1, 0, 2)) \
            + int_tensor_ijj.transpose((1, 2, 0)) + int_tensor_ijj.transpose((2, 1, 0))
        int_tensor_jii_sym = int_tensor_jii + int_tensor_jii.transpose((0, 2, 1)) \
            + int_tensor_iij.transpose((2, 0, 1)) + int_tensor_iij.transpose((2, 1, 0))
        int_tensor_jji_sym_hc = np.conj(int_tensor_jji_sym[::-1, ::-1, ::-1])
        int_tensor_jii_sym_hc = np.conj(int_tensor_jii_sym[::-1, ::-1, ::-1])
        assert np.allclose(int_tensor_jji_sym_hc, int_tensor_jji_sym)
        assert np.allclose(int_tensor_jii_sym_hc, int_tensor_jii_sym)
        # check magnitude of matrix elements
        assert np.allclose(
            np.abs(int_tensor_ijj), np.abs(expected_int_tensor_ijj))
        assert np.allclose(
            np.abs(int_tensor_iij), np.abs(expected_int_tensor_iij))
        assert np.allclose(
            np.abs(int_tensor_jji), np.abs(expected_int_tensor_jji))
        assert np.allclose(
            np.abs(int_tensor_jii), np.abs(expected_int_tensor_jii))
        # check if they're the negative of the corresponding -D interactions
        assert np.allclose(
            int_tensor_ijj, -verts_real_space[3*6 + n+2].interaction_tensor)
        assert np.allclose(
            int_tensor_iij, -verts_real_space[3*6 + n+3].interaction_tensor)
        assert np.allclose(
            int_tensor_jji, -verts_real_space[3*6 + n+4].interaction_tensor)
        assert np.allclose(
            int_tensor_jii, -verts_real_space[3*6 + n+5].interaction_tensor)
        


    
    # FIXED-KS MOMENTUM SPACE
    np.random.seed(2)
    ks = np.array([[np.random.rand(), np.random.rand()],
                   [np.random.rand(), np.random.rand()]])
    # ks = np.array([[1, 0], [1, 0]])
    ks_conserved = np.array([-ks[0]-ks[1], *ks])
    vert_mom_space = momentum_space.compute_magnon_Hamiltonian(
        mod, ks_conserved
    )

    def get_expected_cubic_vert(ks):
        return D[2] * 1j/np.sqrt(2) * sum([
            -np.exp(1j*np.dot(np.sum(ks, axis=0), nnn)) \
            for nnn in np.array([
                [-np.sqrt(3)/2, 3/2], [-np.sqrt(3)/2, -3/2], [np.sqrt(3), 0]
            ])]
        )
    expected_aca_vert = get_expected_cubic_vert(ks)
    expected_caa_vert = get_expected_cubic_vert(ks[1:2])
    expected_cac_vert = get_expected_cubic_vert(-ks[1:2])
    expected_cca_vert = get_expected_cubic_vert(-ks)

    expected_vert_mom_space = np.zeros((4, 4, 4), dtype=complex)
    expected_vert_mom_space[0, 1, 0] = expected_aca_vert
    expected_vert_mom_space[1, 0, 0] = expected_caa_vert
    expected_vert_mom_space[1, 0, 1] = expected_cac_vert
    expected_vert_mom_space[1, 1, 0] = expected_cca_vert
    expected_vert_mom_space[2, 3, 2] = -expected_aca_vert
    expected_vert_mom_space[3, 2, 2] = -expected_caa_vert
    expected_vert_mom_space[3, 2, 3] = -expected_cac_vert
    expected_vert_mom_space[3, 3, 2] = -expected_cca_vert
    # check hermiticity (does not yet work bc vertices are not symmetrized)
    # assert np.allclose(
    #     np.transpose(vert_mom_space, [2, 1, 0])[
    #         np.r_[1, 0, 3, 2], np.r_[1, 0, 3, 2], np.r_[1, 0, 3, 2]
    #     ], np.conj(vert_mom_space))
    # check absolute values
    assert np.allclose(np.abs(vert_mom_space), np.abs(expected_vert_mom_space))

    # FIXED-KS LSWT EIGENSPACE
    _, eigvs = [None]*len(ks_conserved), [None]*len(ks_conserved)
    for n, k in enumerate(ks_conserved):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_magnon_Hamiltonian(
        eigvs, vert_mom_space,
    )
    assert np.allclose(vert_eigenspace, np.einsum(
        "im,jn,kr,ijk", eigvs[0], eigvs[1], eigvs[2], vert_mom_space
    ))



    # BZ MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_mom_space_loop = \
        momentum_space.compute_magnon_Hamiltonians_with_momentum_conservation_and_permutations(
            mod, Momenta(momenta_BZ, k))
    
    
    # BZ EIGENSPACE
    _, eigvs_loop = LSWT.get_eigensystems_momentum_space(
        mod, Momenta.join(
            Momenta.of_BZ(mod.lattice, N_BZ, trans=lambda q: -k-q), 
            Momenta.of_BZ(mod.lattice, N_BZ), 
            Momenta(k)
        ),
    )
    verts_eigenspace_loop = \
        eigenspace.compute_magnon_Hamiltonians_with_permutations(mod, eigvs_loop, verts_mom_space_loop)
    
    for n, p in enumerate(GET_CUBIC_PERMUTATIONS()):
        assert np.allclose(verts_eigenspace_loop.raw_quantity[n], np.einsum(
            "xyim,xyjn,xykr,xyijk->xymnr",
            *permute([
                eigvs_loop.raw_quantity[0], eigvs_loop.raw_quantity[1],
                eigvs_loop.raw_quantity[2][np.newaxis, np.newaxis]
            ], p),
            verts_mom_space_loop.raw_quantity[n]
        ))





def test_normal_order_and_symmetrize_one_band_cubic_vertex():
    vertices = np.arange(48).reshape((6, 1, 2, 2, 2)) + 1
    vertices[0, 0, 1, 0, 0] += 990
    vertices[2, 0, 0, 1, 1] += 8800
    expected_nosym_vertices = np.array([
        (1+9+17+25+33+41)/6,    # a_{-k-q}  a_{q}    a_{k}
        (2+11+18+27+37+45)/2,   # a_{-k-q}  a_{q}    a^†_{-k}
        (3+10+21+29+34+43)/2,   # a_{-k-q}  a^†_{-q} a_{k}
        (4+12+22+31+38+47)/2,   # a_{-k-q}  a^†_{-q} a^†_{-k}
        (995+13+19+26+35+42)/2, # a^†_{k+q} a_{q}    a_{k}
        (6+15+8820+28+39+46)/2, # a^†_{k+q} a_{q}    a^†_{-k}
        (7+14+23+30+36+44)/2,   # a^†_{k+q} a^†_{-q} a_{k}
        (8+16+24+32+40+48)/6,   # a^†_{k+q} a^†_{-q} a^†_{-k}
    ]).reshape((8, 1, 1, 1, 1))

    nosym_vertices = eigenspace_old.normal_order_and_symmetrize_cubic_interaction_Hamiltonian_loop_jit(vertices)

    assert np.allclose(expected_nosym_vertices, nosym_vertices)


def test_normal_order_and_symmetrize_quartic_vertex():
    latt = ChainLattice(2)
    inter = [
        NthNearestNeighborHeisenbergInteraction(latt, 1, J=1),
        MagneticField(latt, 0, np.array([0, 0, 0.1])),
        MagneticField(latt, 1, np.array([0, 0, -0.1])),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, -1]]))

    # REAL SPACE
    verts_real_space = real_space.compute_magnon_Hamiltonian(mod, 4)
    vert_aabb = verts_real_space[9]    # from the S^z S^z term
    
    assert np.allclose(
        np.array([site.bravais_coords for site in vert_aabb.sites]),
        np.array([[0], [0], [-1], [-1]]),
    )
    assert [site.subl_idx for site in vert_aabb.sites] == [0, 0, 1, 1]
    assert np.allclose(
        vert_aabb.interaction_tensor,
        np.array([
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ], [
                [[0, 0], [-1., 0]],
                [[0, 0], [0, 0]],
            ],
        ]),
    )

    # MOMENTUM SPACE
    np.random.seed(2)
    ks = np.random.rand(3, 1)   # q, p, k
    vert_aabb_mom_space = lambda ks: momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation_and_permutations(
        mod, ks, interaction_Hamiltonian_real_space=[vert_aabb],
    )

    expected_vert_aabb_mom_space_base = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    expected_vert_aabb_mom_space_base[1, 0, 3, 2] = -1.
    expected_vert_aabb_mom_space = lambda q, p, k: np.einsum(
        "i,...->i...",
        np.exp(-1j*np.array([
            p+k,    # -k-p-q, q, p, k
            k+p,    # -k-p-q, q, k, p
            q+k,    # -k-p-q, p, q, k
            k+q,    # -k-p-q, p, k, q
            q+p,    # -k-p-q, k, q, p
            p+q,    # -k-p-q, k, p, q
            p+k,    # q, -k-p-q, p, k
            k+p,    # q, -k-p-q, k, p
            (-k-p-q)+k,    # q, p, -k-p-q, k
            k+(-k-p-q),    # q, p, k, -k-p-q
            (-k-p-q)+p,    # q, k, -k-p-q, p
            p+(-k-p-q),    # q, k, p, -k-p-q
            q+k,    # p, -k-p-q, q, k
            k+q,    # p, -k-p-q, k, q
            (-k-p-q)+k,    # p, q, -k-p-q, k
            k+(-k-p-q),    # p, q, k, -k-p-q
            (-k-p-q)+q,    # p, k, -k-p-q, q
            q+(-k-p-q),    # p, k, q, -k-p-q
            q+p,    # k, -k-p-q, q, p
            p+q,    # k, -k-p-q, p, q
            (-k-p-q)+p,    # k, q, -k-p-q, p
            p+(-k-p-q),    # k, q, p, -k-p-q
            (-k-p-q)+q,    # k, p, -k-p-q, q
            q+(-k-p-q),    # k, p, q, -k-p-q
        ])),
        expected_vert_aabb_mom_space_base,
    )
    assert np.allclose(vert_aabb_mom_space(ks), expected_vert_aabb_mom_space(*ks[:, 0]))

    # EIGENSPACE
    vert_aabb_eigenspace = lambda ks, eigvs: eigenspace.compute_magnon_Hamiltonian(
        eigvs, vert_aabb_mom_space(ks)[0b0000],
    )
    
    # COMMUTATOR TERMS EIGENSPACE
    N_BZ = 10
    ks_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell((N_BZ,), as_meshgrid=False)
    eigws_BZ, eigvs_BZ = LSWT.get_eigensystems_momentum_space(mod, [ks_BZ])
    K = np.random.rand(1)
    eigws_K, eigvs_K = LSWT.get_eigensystem_momentum_space(mod, K)
    eigws_minus_K, eigvs_minus_K = LSWT.get_eigensystem_momentum_space(mod, -K)
    vert_aabb_nosym = normal_order.compute_commutator_term_with_permutations(
        mod, [K], [eigvs_minus_K, eigvs_K], ks_BZ, eigvs_BZ[0],
        interaction_Hamiltonian_real_space=[vert_aabb],
    )
    vert_aabb_nosym_HF = normal_order.compute_commutator_term_with_permutations_Hartree_Fock(
        mod, [K], [eigvs_minus_K, eigvs_K], ks_BZ, eigvs_BZ[0],
        interaction_Hamiltonian_real_space=[vert_aabb],
    )
    assert np.allclose(vert_aabb_nosym, vert_aabb_nosym_HF)

    H, P = 0, 1
    sigma_x_ph = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    eigvs = lambda k: LSWT.get_eigensystem_momentum_space(mod, k)[1]
    eigvs_minus = lambda k: sigma_x_ph @ eigvs(k).conj() @ sigma_x_ph
    expected_vert_aabb_nosym = np.sum([
        np.trace(vert_aabb_eigenspace(
            [-K, K, p],
            [eigvs_minus(p), eigvs_minus_K, eigvs_K, eigvs(p)],
        )[H::2, P::2, H::2, P::2], axis1=0, axis2=3) +
        np.trace(vert_aabb_eigenspace(
            [K, -p, p],
            [eigvs_minus_K, eigvs_K, eigvs_minus(p), eigvs(p)],
        )[P::2, H::2, H::2, P::2], axis1=2, axis2=3) +
        np.trace(vert_aabb_eigenspace(
            [-p, K, p],
            [eigvs_minus_K, eigvs_minus(p), eigvs_K, eigvs(p)],
        )[P::2, H::2, H::2, P::2], axis1=1, axis2=3) +
        np.trace(vert_aabb_eigenspace(
            [p, -K, K],
            [eigvs_minus(p), eigvs(p), eigvs_minus_K, eigvs_K],
        )[H::2, P::2, P::2, H::2], axis1=0, axis2=1) +
        np.trace(vert_aabb_eigenspace(
            [-K, p, K],
            [eigvs_minus(p), eigvs_minus_K, eigvs(p), eigvs_K],
        )[H::2, P::2, P::2, H::2], axis1=0, axis2=2) +
        np.trace(vert_aabb_eigenspace(
            [-p, p, K],
            [eigvs_minus_K, eigvs_minus(p), eigvs(p), eigvs_K],
        )[P::2, H::2, P::2, H::2], axis1=1, axis2=2)
        for p in ks_BZ
    ], axis=0)
    
    assert np.allclose(vert_aabb_nosym[0b0, P::2, H::2], expected_vert_aabb_nosym)

    # COMMUTATOR TERMS MOMENTUM SPACE
    verts_aabb_nosym_eigenspace = np.zeros((N_BZ, 2, 4, 4), dtype=np.complex128)
    verts_aabb_nosym_mom_space = np.zeros((N_BZ, 2, 4, 4), dtype=np.complex128)
    for nk, K in enumerate(ks_BZ):
        _, eigvs_K = LSWT.get_eigensystem_momentum_space(mod, K)
        _, eigvs_minus_K = LSWT.get_eigensystem_momentum_space(mod, -K)
        verts_aabb_nosym_eigenspace[nk] = normal_order.compute_commutator_term_with_permutations(
            mod, [K], [eigvs_minus_K, eigvs_K], ks_BZ, eigvs_BZ[0],
            interaction_Hamiltonian_real_space=[vert_aabb],
        )
        verts_aabb_nosym_HF_eigenspace_for_k = normal_order.compute_commutator_term_with_permutations_Hartree_Fock(
            mod, [K], [eigvs_minus_K, eigvs_K], ks_BZ, eigvs_BZ[0],
            interaction_Hamiltonian_real_space=[vert_aabb],
        )
        assert np.allclose(verts_aabb_nosym_eigenspace[nk], verts_aabb_nosym_HF_eigenspace_for_k)

        eigvs_K_inv = np.linalg.inv(eigvs_K)
        eigvs_minus_K_inv = np.linalg.inv(eigvs_minus_K)
        verts_aabb_nosym_mom_space[nk] = np.einsum(
            "pmn,pmi,pnj->pij",
            verts_aabb_nosym_eigenspace[nk],
            [eigvs_minus_K_inv, eigvs_K_inv],
            [eigvs_K_inv, eigvs_minus_K_inv],
        )

    # COMMUTATOR TERMS REAL SPACE
    N_sites = N_BZ
    vert_aabb_nosym_real_space = np.zeros((N_sites, 4, 4), dtype=np.complex128)
    for i in range(N_sites):
        for nk, K in enumerate(ks_BZ):
            vert_aabb_nosym_real_space[i] += \
                1./N_BZ * np.exp(-1j*K.dot(np.array([i]))) * verts_aabb_nosym_mom_space[nk, 0b0]
    
    corr_real_space = compute_real_space_correlator_LSWT(
        [ks_BZ], eigvs_BZ, np.array([[0.], [-1.]]),
    )
    A, A_DAGGER, B, B_DAGGER = range(4)
    expected_vert_aabb_nosym_real_space = np.zeros((N_sites, 4, 4), dtype=np.complex128)
    expected_vert_aabb_nosym_real_space[0, A_DAGGER, A] = -N_BZ * corr_real_space[0, B_DAGGER, B] # decoupling in the a^†a <b^†b> density channel
    expected_vert_aabb_nosym_real_space[0, B_DAGGER, B] = -N_BZ * corr_real_space[0, A_DAGGER, A] # decoupling in the b^†b <a^†a> density channel
    expected_vert_aabb_nosym_real_space[-1, A_DAGGER, B] = -N_BZ * corr_real_space[-1, B_DAGGER, A] # decoupling in the a^†b <b^†a> exchange channel
    expected_vert_aabb_nosym_real_space[-1, B_DAGGER, A] = -N_BZ * corr_real_space[1, A_DAGGER, B] # decoupling in the b^†a <a^†b> exchange channel
    expected_vert_aabb_nosym_real_space[-1, A_DAGGER, B_DAGGER] = -N_BZ * corr_real_space[1, A, B] # decoupling in the a^†ba^† <ab> pairing channel
    expected_vert_aabb_nosym_real_space[-1, A, B] = -N_BZ * corr_real_space[1, A_DAGGER, B_DAGGER] # decoupling in the ab <a^†b^†> pairing channel
    
    # result looks correct when debugging. TODO implement assert
    assert  np.allclose(
        vert_aabb_nosym_real_space,
        expected_vert_aabb_nosym_real_space,
    )