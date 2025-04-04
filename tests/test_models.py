import numpy as np
from magpy.models import Model
from magpy.lattice import BravaisLattice, ChainLattice, HoneycombLatticeA
from magpy.interactions import GammaInteraction, HeisenbergInteraction, KitaevInteraction, UniformMagneticField, NthNearestNeighborHeisenbergInteraction, DMInteraction


def FM_Heisenberg_chain():
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


def AFM_Heisenberg_chain(B=None):
    J = 1.0
    S_A = 3/2
    S_B = 1
    lattice = ChainLattice(2)

    return Model(
        lattice,
        interactions=[
            NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=J),
            *((UniformMagneticField(lattice, B),) if B is not None else ()),
        ],
        classical_ground_state=np.array([[0, 0, S_A], [0, 0, -S_B]])
    ), (J, S_A, S_B)


def FM_Heisenberg_with_DMI_honeycomb(S_A=5/2, S_B=2):
    J = -1.0
    D = 0.1
    D_vec = np.array([0, 0, D])
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



def KH_model_2d():
    #if params is None:
    #    params = {"S": 0.5, "J": -1.53, "K": -29.29, "Gamma": 6.83, "Gamma'": -1.33, "J_3": 0.18, "g_ab": 2.5, "B": 14.0}
    params = {"S": 0.5, "J": -1.53, "K": -29.29, "Gamma": 0, "Gamma'": -1.33, "J_3": 0, "g_ab": 2.5, "B": 50.0}

    S = params["S"]
    J = params["J"]
    J_3 = params["J_3"]
    K = params["K"]
    Gamma = params["Gamma"]
    Gamma_prime = params["Gamma'"]
    B_field_dir = "a"

    g = params["g_ab"] 
    µ = 0.0578  # Bohr magneton in units of meV/Tesla
    B = µ*g*params["B"]*(np.array([-1, 1, 0]/np.sqrt(2) if B_field_dir == "b" else [1, 1, -2]/np.sqrt(6)))
    lattice = HoneycombLatticeA()

    kitaev_order = ("z", "x", "y")
    
    interactions = [
        NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=J),
        # HeisenbergInteraction(BravaisLattice.Edge(np.array([1, 1]), np.array([0, 1])), J_3),
        # HeisenbergInteraction(BravaisLattice.Edge(np.array([-1, 1]), np.array([0, 1])), J_3),
        # HeisenbergInteraction(BravaisLattice.Edge(np.array([1, -1]), np.array([0, 1])), J_3),
        KitaevInteraction(lattice, K, order=kitaev_order),
        # GammaInteraction(lattice, Gamma, order=kitaev_order),
        GammaInteraction(lattice, Gamma_prime, order=kitaev_order, prime=True),
        UniformMagneticField(lattice, B),
    ]

    classical_gs = np.array([-1, 1, 0] if B_field_dir == "b" else [1, 1, -2], dtype=float)
    classical_gs *= S / np.linalg.norm(classical_gs)
    KH_model_2D = Model(lattice, interactions, np.array([classical_gs]*2))

    return KH_model_2D, (S, J, K, Gamma, Gamma_prime, J_3, B)