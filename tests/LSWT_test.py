import numpy as np
from magpy.lattice import DotLattice
from magpy import lattice
from magpy.models import Model
from magpy import models
from magpy import interactions
from magpy.largeS import LSWT


def test_orthonormalization_wrt_bogo_metric_with_Hamiltonian():
    np.random.seed(1)

    for H in [
        np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0.5, 0],
            [0, 0.5, 1, 0],
            [0.5, 0, 0, 1],
        ]), np.array([
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, -1, 0, 1],
        ]), np.array([
            [1, 0.1, 0.1, 0],
            [0.1, 1, 0, 0.1],
            [0.1, 0, 1, 0.1],
            [0, 0.1, 0.1, 1],
        ]), np.array([
            [2, 0, 1, 0, 1, 0],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 1, 0, 2, 0, 1],
            [1, 0, 1, 0, 2, 0],
            [0, 1, 0, 1, 0, 2],
        ]),
    ]:
        N = H.shape[0] // 2
        bogo_metric = np.diag([1, -1]*N)
        model = Model(DotLattice(N), [], np.array([[0, 0, 1]]*N))
        eigw, eigv = LSWT.get_eigensystem_momentum_space(model, LSWT_Hamiltonian_momentum_space_BdG=H)
        # eigv = np.random.rand(2*N, 2*N)
        eigv_ortho = LSWT.orthogonalize_wrt_metric(eigv, bogo_metric)
        eigv_ortho = LSWT.normalize_wrt_metric(eigv_ortho, bogo_metric)

        # test if eigenvectors are orthonormal wrt bogo metric
        assert np.allclose(
            eigv_ortho.conj().T @ bogo_metric @ eigv_ortho,
            bogo_metric)
        # test if eigenvectors fulfill the eigenvalue equation
        assert np.allclose(
            bogo_metric @ H @ eigv_ortho - eigv_ortho @ np.diag(eigw),
            np.zeros((2*N, 2*N)))
        

def test_orthonormalization_wrt_bogo_metric_with_given_eigenvectors():
    np.random.seed(1)

    H0 = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0.5, 0],
            [0, 0.5, 1, 0],
            [0.5, 0, 0, 1],
        ])
    _, eigv0 = LSWT.get_eigensystem_momentum_space(
        Model(DotLattice(2), [], np.array([[0, 0, 1]]*2)),
        LSWT_Hamiltonian_momentum_space_BdG=H0
    )
    # mix up the eigenvectors a bit so that they are not orthogonal anymore
    eigv0[:, 0] += 0.4*eigv0[:, 2]

    for eigv in [
        eigv0
    ]:
        N = eigv.shape[0] // 2
        bogo_metric = np.diag([1, -1]*N)
        eigv_ortho = LSWT.orthogonalize_wrt_metric(eigv, bogo_metric)
        eigv_ortho = LSWT.normalize_wrt_metric(eigv_ortho, bogo_metric)

        assert np.allclose(
            eigv_ortho.conj().T @ bogo_metric @ eigv_ortho,
            bogo_metric)
        

def test_YFeO_CAFM_zone_center_3d():
    J = 1# 4.96
    D = 1.0# 0.11
    Ka = 0.1#0.0046
    Kc = 0.1#.0011
    theta = 0.5 * (np.arctan(-4*D / (6*J + Ka - Kc)) + np.pi)
    classical_gs = 5/2 * np.array(
        [[np.sin(theta), 0, np.cos(theta)],
        [-np.sin(theta), 0, np.cos(theta)]]
    )

    latt_2d = lattice.BravaisLattice(np.eye(2), np.array([[0.5, 0], [0, 0.5]]), [
        lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([1, 0])),
        lattice.BravaisLattice.Edge(np.array([-1, 1]), np.array([1, 0])),
    ], {})
    mod_2d = models.Model(
        latt_2d, interactions=[
            interactions.NthNearestNeighborHeisenbergInteraction(latt_2d, n=1, J=J),
            interactions.DMInteraction(lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 1])), D=np.array([0, D, 0])),
            interactions.DMInteraction(lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])), D=np.array([0, D, 0])),
            interactions.DMInteraction(lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([1, 0])), D=np.array([0, -D, 0])),
            interactions.DMInteraction(lattice.BravaisLattice.Edge(np.array([-1, 1]), np.array([1, 0])), D=np.array([0, -D, 0])),
            interactions.Interaction([
                lattice.BravaisLattice.Site(np.array([0, 0]), np.array([0])),
                lattice.BravaisLattice.Site(np.array([0, 0]), np.array([0])),
            ], np.diag([-Ka, 0, -Kc])),
            interactions.Interaction([
                lattice.BravaisLattice.Site(np.array([0, 0]), np.array([1])),
                lattice.BravaisLattice.Site(np.array([0, 0]), np.array([1])),
            ], np.diag([-Ka, 0, -Kc])),
        ], classical_ground_state=classical_gs,
    )
    
    theta_3d = 0.5 * (np.arctan(-4*D / (4*J + Ka - Kc)) + np.pi)
    interlayer_edges = [
        lattice.BravaisLattice.Edge(np.array([0, -1, 1]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([-1, 0, 1]), np.array([1, 0])),
    ]
    mod_3d = models.stack(
        mod_2d,
        num_layers=1,
        interlayer_edges=interlayer_edges,
        interlayer_interactions=[
            interactions.HeisenbergInteraction(edge, J=J) for edge in interlayer_edges
        ],
        new_classical_ground_state=5/2 * np.array(
            [[np.sin(theta_3d), 0, np.cos(theta_3d)],
            [-np.sin(theta_3d), 0, np.cos(theta_3d)]]
        ),
        periodic=True,
        additional_bravais_vec=np.array([0.5, 0.5, 1]),
    )

    np.set_printoptions(precision=4, suppress=True)

    A = -(6*J)*np.cos(2*theta_3d) + 4*D*np.sin(2*theta_3d) + 2*(Ka*np.sin(theta_3d)**2 + Kc*np.cos(theta_3d)**2) - (Ka*np.cos(theta_3d)**2 + Kc*np.sin(theta_3d)**2)
    B = -(Ka*np.cos(theta_3d)**2 + Kc*np.sin(theta_3d)**2)
    h_plus = 0.5*(J*(np.cos(2*theta_3d) + 1) - D*np.sin(2*theta_3d)) * 4 + 0.5*J*(np.cos(2*theta_3d) + 1) * 2
    h_minus = 0.5*(J*(np.cos(2*theta_3d) - 1) - D*np.sin(2*theta_3d)) * 4 + 0.5*J*(np.cos(2*theta_3d) - 1) * 2
    expected_H = 5/2 * np.array([
        [A, B, -h_plus, -h_minus],
        [B, A, -h_minus, -h_plus],
        [-h_plus, -h_minus, A, B],
        [-h_minus, -h_plus, B, A],
    ])

    H = LSWT.compute_LSWT_Hamiltonian_momentum_space_BdG(mod_3d, np.zeros(3))
    assert np.allclose(H, expected_H)
    
