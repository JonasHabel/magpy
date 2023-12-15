from magpy.lattice import *
from magpy import models
from magpy.interactions import *
from magpy.interaction_vertices import real_space, momentum_space, eigenspace
from magpy.interaction_vertices.util import GET_CUBIC_PERMUTATIONS
from magpy import LSWT
import numpy as np
from magpy.util import permute


def test_two_site_quantum_dot_with_DMI():
    latt = DotLattice(2)
    mod = models.Model(latt, [
        DMInteraction(BravaisLattice.Edge(np.array([]), [0, 1]), D=np.array([1.0, 0, 0]))
    ], classical_ground_state=np.array([[0, 0, 1], [0, 0, 1]]))

    # REAL SPACE
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)

    # expect -1/sqrt(2) [a_2^† a_1 a_2 - a_1^† a_1 a_2 + h.c.] 
    #       - 1/sqrt(8) [a_1^† a_1 a_1 - a_2^† a_2 a_2 + h.c.]
    expected_int_tensor_111 = -1/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [-1j, 0]],
    ])
    expected_int_tensor_222 = 1/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [-1j, 0]],
    ])
    expected_int_tensor_212 = -1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [0, 0]],
    ])
    expected_int_tensor_122 = -1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [-1j, 0]],
    ])
    expected_int_tensor_112 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[1j, 0], [0, 0]],
    ])
    expected_int_tensor_121 = 1/(np.sqrt(2)) * np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [-1j, 0]],
    ])

    assert len(verts_real_space) == 6

    assert np.allclose(expected_int_tensor_111, verts_real_space[0].interaction_tensor)
    assert np.allclose(expected_int_tensor_222, verts_real_space[1].interaction_tensor)
    assert np.allclose(expected_int_tensor_212, verts_real_space[2].interaction_tensor)
    assert np.allclose(expected_int_tensor_122, verts_real_space[3].interaction_tensor)
    assert np.allclose(expected_int_tensor_112, verts_real_space[4].interaction_tensor)
    assert np.allclose(expected_int_tensor_121, verts_real_space[5].interaction_tensor)

    # MOMENTUM SPACE
    vert_mom_space = momentum_space.compute_interaction_Hamiltonian(
        mod, 3, np.array([[]]), verts_real_space)
    
    expected_vert_mom_space = 1/(2*np.sqrt(8)) * np.array([
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[-1j, 0, 4j, 0], [1j, 0, 0, 0], [0, 0, 0, 0], [-4j, 0, 4j, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, -4j, 0], [0, 0, 0, 0], [0, 0, 1j, 0], [0, 0, -1j, 0]],
    ])

    assert np.allclose(vert_mom_space, expected_vert_mom_space)

    # LSWT EIGENSPACE
    _, eigvs = LSWT.get_eigensystem_momentum_space(mod, np.array([0]))
    eigvs = np.array([eigvs, eigvs, eigvs])
    
    vert_eigenspace = eigenspace.compute_interaction_Hamiltonian(
        mod, 3, eigvs, vert_mom_space)
    
    assert np.allclose(vert_eigenspace, expected_vert_mom_space)



def test_field_orthogonal_to_quantization_direction():
    latt = SquareLattice()
    B = np.array([1, 2, 3])
    inter = [
        MagneticField(latt, 0, B),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))

    # REAL SPACE
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)
    
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
    vert_mom_space = momentum_space.compute_interaction_Hamiltonian(
        mod, 3, ks, verts_real_space)
    # in this testcase, the real space and momentum space vertex coefficients
    # should be the same
    assert np.allclose(vert_mom_space, int_tensor)

    # FIXED-KS LSWT EIGENSPACE
    eigvs = [None]*len(ks)
    for n, k in enumerate(ks):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_interaction_Hamiltonian(
        mod, 3, eigvs, vert_mom_space
    )
    assert np.allclose(vert_eigenspace, np.einsum(
        "im,jn,kr,ijk", eigvs[0], eigvs[1], eigvs[2], vert_mom_space
    ))

    # BZ MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_for_loop_momentum = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ)
    
    assert np.allclose(verts_for_loop_momentum,
                       vert_mom_space[np.newaxis, np.newaxis, np.newaxis, ...])
    
    # BZ EIGENSPACE
    _, eigvs_at_k = LSWT.get_eigensystem_momentum_space(mod, k)
    _, eigvs_BZ = LSWT.get_eigensystem_for_Brillouin_zone(mod, N_BZ)
    _, eigvs_minus_k_minus_BZ = LSWT.get_eigensystem_for_loop_momentum(
        mod, -k, N_BZ)
    verts_eigenspace_for_loop_momentum = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
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
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)
    # FM interaction does not yield any cubic terms
    assert all(map(
        lambda vert: np.all(vert.interaction_tensor == 0),
        verts_real_space))
    
    # FIXED-KS MOMENTUM SPACE
    ks = np.array([[np.random.rand(), np.random.rand()],
                   [np.random.rand(), np.random.rand()]])
    ks = np.array([*ks, -ks[0]-ks[1]])
    vert_mom_space = momentum_space.compute_interaction_Hamiltonian(
        mod, 3, ks, verts_real_space)
    assert np.all(vert_mom_space == 0)

    # FIXED-KS LSWT EIGENSPACE
    eigvs = [None]*len(ks)
    for n, k in enumerate(ks):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_interaction_Hamiltonian(
        mod, 3, eigvs, vert_mom_space
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
    verts_real_space = real_space.compute_interaction_Hamiltonian(mod, order=3)



    assert len(verts_real_space) == 3*6 + 6*6
    # FM interaction does not yield any cubic terms
    assert all(map(
        lambda vert: np.all(vert.interaction_tensor == 0),
        verts_real_space[:3*6]))
    

    # DMI does yield cubic terms
    # first, the iii- and jjj-terms (each up to gauge choice)
    expected_int_tensor_iii = D[2] * 1j/(2*np.sqrt(8)) * np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [-1, 0]],
    ])
    expected_int_tensor_jjj = -expected_int_tensor_iii
    for n in (3*6 + 6*np.arange(3)):
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
        
    # next, the iij-, iji-, jij- and ijj-terms (each up to gauge choice)
    expected_int_tensor_iij = D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [0, 0]],
    ])
    expected_int_tensor_iji = -D[2] * 1j/np.sqrt(2) * np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ])
    expected_int_tensor_jij = -expected_int_tensor_iij
    expected_int_tensor_ijj = -expected_int_tensor_iji
    for n in (3*6 + 6*np.arange(3)):
        expected_subl_idx = 0
        for x in range(2, 6):
            assert all(map(
                lambda site: site.subl_idx == expected_subl_idx,
                verts_real_space[n+x].sites))
        int_tensor_jij = verts_real_space[n+2].interaction_tensor
        int_tensor_ijj = verts_real_space[n+3].interaction_tensor
        int_tensor_iij = verts_real_space[n+4].interaction_tensor
        int_tensor_iji = verts_real_space[n+5].interaction_tensor
        # check hermiticity
        assert np.allclose(int_tensor_jij[:, ::-1, :], np.conj(int_tensor_ijj))
        assert np.allclose(int_tensor_iij[:, ::-1, :], np.conj(int_tensor_iji))
        # check magnitude of matrix elements
        assert np.allclose(
            np.abs(int_tensor_jij), np.abs(expected_int_tensor_jij))
        assert np.allclose(
            np.abs(int_tensor_ijj), np.abs(expected_int_tensor_ijj))
        assert np.allclose(
            np.abs(int_tensor_iij), np.abs(expected_int_tensor_iij))
        assert np.allclose(
            np.abs(int_tensor_iji), np.abs(expected_int_tensor_iji))
        # check if they're the negative of the corresponding -D interactions
        assert np.allclose(
            int_tensor_jij, -verts_real_space[3*6 + n+2].interaction_tensor)
        assert np.allclose(
            int_tensor_ijj, -verts_real_space[3*6 + n+3].interaction_tensor)
        assert np.allclose(
            int_tensor_iij, -verts_real_space[3*6 + n+4].interaction_tensor)
        assert np.allclose(
            int_tensor_iji, -verts_real_space[3*6 + n+5].interaction_tensor)
        


    
    # FIXED-KS MOMENTUM SPACE
    ks = np.array([[np.random.rand(), np.random.rand()],
                   [np.random.rand(), np.random.rand()]])
    # ks = np.array([[1, 0], [1, 0]])
    ks = np.array([*ks, -ks[0]-ks[1]])
    vert_mom_space = momentum_space.compute_interaction_Hamiltonian(
        mod, 3, ks, verts_real_space)
    expected_caa_vert = D[2] * 1j/np.sqrt(2) * sum([
        np.exp(1j*np.dot(ks[0] + ks[1], nnn)) - np.exp(1j*np.dot(ks[1], nnn)) \
        for nnn in np.array([
            [-np.sqrt(3)/2, 3/2], [-np.sqrt(3)/2, -3/2], [np.sqrt(3), 0]
        ])]
    )
    expected_cca_vert = D[2] * 1j/np.sqrt(2) * sum([
        np.exp(-1j*np.dot(ks[0], nnn)) - np.exp(1j*np.dot(ks[1], nnn)) \
        for nnn in np.array([
            [-np.sqrt(3)/2, 3/2], [-np.sqrt(3)/2, -3/2], [np.sqrt(3), 0]
        ])]
    )
    expected_vert_mom_space = np.zeros((4, 4, 4), dtype=complex)
    expected_vert_mom_space[1, 0, 0] = expected_caa_vert
    expected_vert_mom_space[1, 1, 0] = expected_cca_vert
    expected_vert_mom_space[3, 2, 2] = -expected_caa_vert
    expected_vert_mom_space[3, 3, 2] = -expected_cca_vert
    # check hermiticity (does not yet work bc vertices are not symmetrized)
    # assert np.allclose(
    #     np.transpose(vert_mom_space, [2, 1, 0])[
    #         np.r_[1, 0, 3, 2], np.r_[1, 0, 3, 2], np.r_[1, 0, 3, 2]
    #     ], np.conj(vert_mom_space))
    # check absolute values
    assert np.allclose(np.abs(vert_mom_space), np.abs(expected_vert_mom_space))

    # FIXED-KS LSWT EIGENSPACE
    _, eigvs = [None]*len(ks), [None]*len(ks)
    for n, k in enumerate(ks):
        _, eigvs[n] = LSWT.get_eigensystem_momentum_space(mod, k)
    eigvs = np.array(eigvs)
    vert_eigenspace = eigenspace.compute_interaction_Hamiltonian(
        mod, 3, eigvs, vert_mom_space
    )
    assert np.allclose(vert_eigenspace, np.einsum(
        "im,jn,kr,ijk", eigvs[0], eigvs[1], eigvs[2], vert_mom_space
    ))



    # BZ MOMENTUM SPACE
    k = np.array([np.random.rand(), np.random.rand()])
    N_BZ = (5, 5)
    momenta_BZ = mod.lattice.reciprocal_lattice.sample_inverse_unit_cell(
        N_BZ).reshape((*N_BZ, 2))
    verts_for_loop_momentum = \
        momentum_space.compute_cubic_interaction_Hamiltonian_loop(
            mod, k, momenta_BZ)
    
    
    # BZ EIGENSPACE
    _, eigvs_at_k = LSWT.get_eigensystem_momentum_space(mod, k)
    _, eigvs_BZ = LSWT.get_eigensystem_for_Brillouin_zone(mod, N_BZ)
    _, eigvs_minus_k_minus_BZ = LSWT.get_eigensystem_for_loop_momentum(
        mod, -k, N_BZ)
    verts_eigenspace_for_loop_momentum = \
        eigenspace.compute_cubic_interaction_Hamiltonian_loop(
            mod, eigvs_at_k, eigvs_BZ, eigvs_minus_k_minus_BZ, verts_for_loop_momentum)
    
    for n, p in enumerate(GET_CUBIC_PERMUTATIONS()):
        assert np.allclose(verts_eigenspace_for_loop_momentum[n], np.einsum(
            "xyim,xyjn,xykr,xyijk->xymnr",
            *permute([
                eigvs_minus_k_minus_BZ, eigvs_BZ, 
                eigvs_at_k[np.newaxis, np.newaxis]
            ], p),
            verts_for_loop_momentum[n]
        ))