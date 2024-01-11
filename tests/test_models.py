import numpy as np
from magpy.models import Model
from magpy.lattice import BravaisLattice, ChainLattice, HoneycombLatticeA
from magpy.interactions import NthNearestNeighborHeisenbergInteraction, DMInteraction


def FM_Heisenberg_chain():
    J = 1.0
    S_A = 3/2
    S_B = 1
    lattice = ChainLattice(2)

    return Model(
        lattice,
        interactions=[
            NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=J),
        ],
        classical_ground_state=np.array([[0, 0, S_A], [0, 0, -S_B]])
    ), (J, S_A, S_B)


def AFM_Heisenberg_chain():
    J = -1.0
    S = 3/2
    lattice = ChainLattice()

    return Model(
        lattice,
        interactions=[
            NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=J),
        ],
        classical_ground_state=S*np.array([[0, 0, 1]])
    ), (J, S)


def FM_Heisenberg_with_DMI_honeycomb():
    J = -1.0
    D = 0.1
    D_vec = np.array([0, 0, D])
    S_A = 5/2
    S_B = 2
    theta = 0.1 * np.pi/2
    lattice = HoneycombLatticeA()

    return Model(
        lattice,
        interactions=[
            NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=J),
            DMInteraction(BravaisLattice.Edge(np.array([1, -1]), np.array([0, 0])), D=D_vec),
            DMInteraction(BravaisLattice.Edge(np.array([-1, 0]), np.array([0, 0])), D=D_vec),
            DMInteraction(BravaisLattice.Edge(np.array([0, 1]), np.array([0, 0])), D=D_vec),
            DMInteraction(BravaisLattice.Edge(np.array([1, -1]), np.array([1, 1])), D=-D_vec),
            DMInteraction(BravaisLattice.Edge(np.array([-1, 0]), np.array([1, 1])), D=-D_vec),
            DMInteraction(BravaisLattice.Edge(np.array([0, 1]), np.array([1, 1])), D=-D_vec),
        ],
        classical_ground_state=np.array([
            [S_A*np.sin(theta), 0, S_A*np.cos(theta)],
            [S_B*np.sin(theta), 0, S_B*np.cos(theta)],
        ])
    ), (J, D, S_A, S_B, theta)
