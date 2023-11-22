import numpy as np
from magpy import models, lattice, interactions, LLG


def test_quantum_dot():
    latt = lattice.DotLattice(2)
    inter = [
        interactions.HeisenbergInteraction(
            edge=lattice.BravaisLattice.Edge(np.array([]), np.array([0, 1])),
            J=-1.0),
        interactions.DMInteraction(
            edge=lattice.BravaisLattice.Edge(np.array([]), np.array([0, 1])),
            D=np.array([0, 0.5, 0])),
        interactions.MagneticField(latt, 0, B=np.array([0, 3.0, 0])),
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, 1]]))
    
    spin_config = np.array([[0, 0, 1], [0, 0, 1]])
    eff_field = LLG.compute_effective_field(mod, spin_config,
                                            np.array([], dtype=int),
                                            len(latt.sublattices), use_jit=False)
    assert np.allclose(eff_field, np.array([[0.5, 3.0, 1], [-0.5, 0, 1]]))

    spin_config = np.array([[0, 0, 1], [1, 0, 0]])
    eff_field = LLG.compute_effective_field(mod, spin_config,
                                            np.array([], dtype=int),
                                            len(latt.sublattices), use_jit=True)
    assert np.allclose(eff_field, np.array([[1, 3.0, -0.5], [-0.5, 0, 1]]))


def test_FM_Heisenberg_chain():
    latt = lattice.ChainLattice()
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=-1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1]]))
    dim = 10

    spin_config = np.array([[[0, 0, 1]]]*dim) \
        .reshape((dim*len(latt.sublattices), 3))
    eff_field = LLG.compute_effective_field(mod, spin_config, np.array([dim]),
                                            len(latt.sublattices))
    assert np.allclose(eff_field, np.array(
        [[[0, 0, 2]]]*dim
    ).reshape((dim*len(latt.sublattices), 3)))

    spin_config = np.array([[[1, 0, 0]]] + [[[0, 0, 1]]]*(dim - 1)) \
        .reshape((dim*len(latt.sublattices), 3))
    eff_field = LLG.compute_effective_field(mod, spin_config, np.array([dim]),
                                            len(latt.sublattices))
    assert np.allclose(eff_field, np.array(
        [[[0, 0, 2]]] \
        + [[[1, 0, 1]]] \
        + [[[0, 0, 2]]]*(dim-3) + [[[1, 0, 1]]]
    ).reshape((dim*len(latt.sublattices), 3)))


def test_AFM_Heisenberg_chain():
    latt = lattice.ChainLattice(2)
    inter = [
        interactions.NthNearestNeighborHeisenbergInteraction(
            latt, n=1, J=1.0)
    ]
    mod = models.Model(latt, inter, np.array([[0, 0, 1], [0, 0, -1]]))
    dim = 10

    spin_config = np.array([[[0, 0, 1], [0, 0, 1]]]*dim) \
        .reshape((dim*len(latt.sublattices), 3))
    eff_field = LLG.compute_effective_field(mod, spin_config, np.array([dim]),
                                            len(latt.sublattices))
    assert np.allclose(eff_field, np.array(
        [[[0, 0, -2], [0, 0, -2]]]*dim
    ).reshape((dim*len(latt.sublattices), 3)))

    spin_config = np.array([[[0, 0, 1], [1, 0, 0]]] + [[[0, 0, 1], [0, 0, 1]]]*(dim-1)) \
        .reshape((dim*len(latt.sublattices), 3))
    eff_field = LLG.compute_effective_field(mod, spin_config, np.array([dim]),
                                            len(latt.sublattices))
    assert np.allclose(eff_field, np.array(
        [[[-1, 0, -1], [0, 0, -2]], [[-1, 0, -1], [0, 0, -2]]] \
        + [[[0, 0, -2], [0, 0, -2]]]*(dim-2)
    ).reshape((dim*len(latt.sublattices), 3)))


def test_YFeO_CAFM_ground_state():
    J = 1# 4.96
    D = 1.0# 0.11
    Ka = 0.1#0.0046
    Kc = 0.1#.0011
    theta = 0.5 * (np.arctan(-4*D / (4*J + Ka - Kc)) + np.pi)
    classical_gs = 5/2 * np.array(
        [[np.sin(theta), 0, np.cos(theta)],
        [-np.sin(theta), 0, np.cos(theta)]]
    )

    latt = lattice.BravaisLattice(np.eye(2), np.array([[0.5, 0], [0, 0.5]]), [
        lattice.BravaisLattice.Edge(np.array([1, 0]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
        lattice.BravaisLattice.Edge(np.array([0, 1]), np.array([1, 0])),
        lattice.BravaisLattice.Edge(np.array([-1, 1]), np.array([1, 0])),
    ], {})
    mod = models.Model(
        latt, interactions=[
            interactions.NthNearestNeighborHeisenbergInteraction(latt, n=1, J=J),
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

    dims = 1, 1
    gs_spin_config = (np.ones((*dims, 2, 3)) * classical_gs[np.newaxis, np.newaxis, :, :]) \
        .reshape((np.prod(dims)*2, 3))
    eff_field = LLG.compute_effective_field(mod, gs_spin_config, np.array(dims), 2, use_jit=False)
    eff_field_normalized = eff_field / np.linalg.norm(eff_field, axis=1)[:, np.newaxis]
    gs_spin_config_normalized = gs_spin_config / np.linalg.norm(gs_spin_config, axis=1)[:, np.newaxis]

    # in the ground state, the effective field some site should point along the
    # spin at that site
    assert np.allclose(eff_field_normalized, gs_spin_config_normalized)
    



