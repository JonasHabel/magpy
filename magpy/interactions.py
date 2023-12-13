import numpy as np
from .lattice import BravaisLattice, HoneycombLatticeA, HoneycombLatticeB
from . import util
from .util import LEVI_CIVITA
from collections import deque

class Interaction:
    """
    sites: list of BravaisLattice.Site
    interaction_tensor: (3)**(len(sites)) numpy array 
    """
    def __init__(self, sites, interaction_tensor):
        self.sites = sites
        self.interaction_tensor = interaction_tensor

    def copy(self):
        return Interaction(
            [
                BravaisLattice.Site(site.bravais_coords, site.subl_idx) \
                for site in self.sites
            ],
            self.interaction_tensor.copy()
        )

    """
    See doc of Model.compute_rotated_interactions
    """
    def rotate_spin_coord_system(self, rot_matrices_for_sublattices):
        num_spins = len(self.sites)
        rot_matrices = np.zeros((num_spins, 3, 3), dtype=complex)
        for i, site in enumerate(self.sites):
            rot_matrices[i] = rot_matrices_for_sublattices[site.subl_idx]
        
        rotated_interaction_tensor = \
            util.tensor_rotate(self.interaction_tensor, rot_matrices)
        
        return Interaction(self.sites, rotated_interaction_tensor)


class MagneticField(Interaction):
    def __init__(self, lattice: BravaisLattice, sublattice_index: int, B):
        super().__init__(
            sites=[BravaisLattice.Site(
                bravais_coords=np.zeros(lattice.dim, dtype=int),
                sublattice_index=sublattice_index
            )],
            interaction_tensor=-B
        )


TWO_SPIN_INT_TENSORS = {
    "Heisenberg": lambda J: J*np.identity(3),
    "Ising": lambda dir, J: J*np.diag(np.identity(3)[dir]),
    "DM": lambda D: np.tensordot(D, LEVI_CIVITA, axes=[[0], [0]]),
}

class TwoSpinInteraction(Interaction):
    def __init__(self, edge: BravaisLattice.Edge, interaction_tensor):
        super().__init__(
            sites=edge.get_sites(),
            interaction_tensor=interaction_tensor
        )


class HeisenbergInteraction(TwoSpinInteraction):
    def __init__(self, edge: BravaisLattice.Edge, J):
        super().__init__(edge, TWO_SPIN_INT_TENSORS["Heisenberg"](J))


class IsingInteraction(TwoSpinInteraction):
    """
    dir: {0, 1, 2} or {"x", "y", "z"}
        direction of Ising anisotropy (0 = x, 1 = y, 2 = z)
    J:
        interaction strength
    """
    def __init__(self, edge: BravaisLattice.Edge, dir, J):
        if type(dir) == str:
            dir = map_dir_to_index(dir)
        super().__init__(edge, TWO_SPIN_INT_TENSORS["Ising"](dir, J))

def map_dir_to_index(dir):
    return {"x": 0, "y": 1, "z": 2}[dir]


class DMInteraction(TwoSpinInteraction):
    def __init__(self, edge: BravaisLattice.Edge, D):
        if len(D.shape) != 1 or D.shape[0] != 3:
            raise Exception("Expected DMI vector of length 3 instead of {D.shape}.")
        super().__init__(edge, TWO_SPIN_INT_TENSORS["DM"](D))


class AnisotropyInteraction(TwoSpinInteraction):
    def __init__(self, sublattice_index: int, dir, A):
        if type(dir) == str:
            dir = map_dir_to_index(dir)
        super().__init__(
            BravaisLattice.Edge(np.array([0, 0]), np.array([sublattice_index]*2)),
            TWO_SPIN_INT_TENSORS["Ising"](dir, A))


class CompositeInteraction():
    def __init__(self, interactions):
        self.interactions = interactions

    def expand(self):
        expanded_inter = []
        for child in self.interactions:
            if isinstance(child, CompositeInteraction):
                expanded_inter += child.expand()
            else:
                expanded_inter.append(child)
                
        return expanded_inter
    

class UniformMagneticField(CompositeInteraction):
    def __init__(self, lattice: BravaisLattice, B):
        super().__init__([
            MagneticField(lattice, subl_idx, B) \
            for subl_idx in range(lattice.num_sites_unit_cell)
        ])


class NthNearestNeighborInteraction(CompositeInteraction):
    def __init__(self, lattice: BravaisLattice, n: int, interaction_tensor):
        nth_nns = lattice.compute_nth_nearest_neighbors_for_entire_unit_cell(n)
        super().__init__([
            TwoSpinInteraction(
                edge=nth_nn, interaction_tensor=interaction_tensor
            ) for nth_nn in nth_nns
        ])


class NthNearestNeighborHeisenbergInteraction(NthNearestNeighborInteraction):
    def __init__(self, lattice: BravaisLattice, n: int, J):
        super().__init__(lattice, n, TWO_SPIN_INT_TENSORS["Heisenberg"](J))


class NthNearestNeighborDMInteraction(NthNearestNeighborInteraction):
    def __init__(self, lattice: BravaisLattice, n: int, D):
        super().__init__(lattice, n, TWO_SPIN_INT_TENSORS["DM"](D))


class NthNearestNeighborIsingInteraction(NthNearestNeighborInteraction):
    def __init__(self, lattice: BravaisLattice, n: int, dir, J):
        if type(dir) == str:
            dir = map_dir_to_index(dir)
        super().__init__(lattice, n, TWO_SPIN_INT_TENSORS["Ising"](dir, J))


class KitaevInteraction(CompositeInteraction):
    def __init__(self, lattice: BravaisLattice, K, order=("x", "z", "y")):
        nns = lattice.compute_nth_nearest_neighbors_for_entire_unit_cell(n=1)
        nns = sorted(nns,
            key=lambda nn: KitaevInteraction.bond_order_key(lattice, nn))
        super().__init__([
            IsingInteraction(nn, dir, K) for nn, dir in zip(nns, order)
        ])

    def bond_order_key(lattice: BravaisLattice, nn):
        nn_canonical_basis = lattice.get_canonical_coords_for_edge(nn)
        return np.angle(nn_canonical_basis[0] + 1j*nn_canonical_basis[1])


class GammaInteraction(CompositeInteraction):
    def __init__(self, lattice: HoneycombLatticeB, Gamma, order=("x", "z", "y"),
                 prime=False):
        nns = lattice.compute_nth_nearest_neighbors_for_entire_unit_cell(n=1)
        nns = sorted(nns, key=lambda nn: nn.bravais_coords.dot(np.ones(2)))
        super().__init__([
            TwoSpinInteraction(nn, self.__get_int_tensor(Gamma, dir, prime)) \
            for nn, dir in zip(nns, order)
        ])

    def __get_int_tensor(self, Gamma, dir, prime):
        if type(dir) == str:
            dir = map_dir_to_index(dir)
            
        int_tensor = np.zeros((3, 3))
        if prime:
            int_tensor[dir, (dir-1)%3] = 1
            int_tensor[dir, (dir+1)%3] = 1
        else:
            int_tensor[(dir-1)%3, (dir+1)%3] = 1

        int_tensor += int_tensor.T
        return Gamma * int_tensor
