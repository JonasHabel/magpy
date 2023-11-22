import numpy as np
from magpy.lattice import DotLattice
from magpy.models import Model
from magpy import LSWT


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
        eigw, eigv = LSWT.get_eigensystem_momentum_space(model, magnon_Hamiltonian=H)
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
        magnon_Hamiltonian=H0
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