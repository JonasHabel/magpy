import numpy as np
from ..magpy import models, lattice, interactions, LLG


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
