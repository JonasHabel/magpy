import numpy as np
from .lattice import BravaisLattice, HoneycombLatticeA, HoneycombLatticeB
from . import util
from .util import LEVI_CIVITA
from collections import Counter

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
    
    def __eq__(self, other):
        assert len(self.sites) == len(other.sites)

        for site, other_site in zip(self.sites, other.sites):
            if site != other_site:
                return False
        
        return np.allclose(self.interaction_tensor, other.interaction_tensor)

    def __hash__(self):
        return hash(tuple([
            *self.sites, 
            tuple([
                len(self.interaction_tensor.shape),
                *self.interaction_tensor.shape,
                *self.interaction_tensor.flatten(),
            ]),
        ]))

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




"""
Associate an "ID" to each interaction such that physically equivalent
interactions have the same ID. Then, merge the interaction tensors of 
interactions with the same ID.

The ID of an interaction is determined by the list of sites on which it acts.
We have some freedom in choosing the function that computes the IDs, as long
as interactions with the same ID are physically equivalent.
We may choose it such that the ID of two interactions a and b is equal IFF
1. a.sites == b.sites, i.e., the sites are exactly the same and in the 
   same order,
2. a.sites is related to b.sites by a permutation (requires appropriate
   transposition of the axes of the interaction tensor),
3. there is a bravais lattice vector V such that moving a.sites[i] by V
   yields b.sites[i] for all i,
4. both 2 and 3 combined, i.e., there is a bravais lattice vector V and a
   permutation p such that moving a.sites[i] by V yields b.sites[p(i)] for all i
The compression ratio increases from 1. through 4.
"""
def compress(interactions, /, permute=False, translate=False):
    if len(interactions) == 0:
        return []

    assert all(map(
        lambda inter: all(map(
            lambda dim: dim == interactions[0].interaction_tensor.shape[0],
            inter.interaction_tensor.shape)), 
        interactions))
    
    # get the Bravais coordinates of the unit cell where the center of mass
    # of sites is located
    def get_center_of_mass_Bravais_offset(sites):
        center = np.sum(np.array([site.bravais_coords for site in sites]), axis=0) / len(sites)
        center_bravais_offset = np.floor(center).astype(np.int64)
        return center_bravais_offset

    def get_id(inter):
        x = tuple(inter.sites)
        if translate and len(inter.sites) >= 1:
            center_bravais_offset = get_center_of_mass_Bravais_offset(x)
            x = tuple(site.translate(-center_bravais_offset) for site in x)
        if permute:
            x = frozenset(Counter(x).items())
        return hash(x)

    # translate sites of inter so that their center of mass is in the same
    # unit cell as the center of mass of the sites of reference_inter
    def get_translated_sites(inter, reference_inter):
        center_bravais_offset = get_center_of_mass_Bravais_offset(inter.sites)
        ref_center_bravais_offset = get_center_of_mass_Bravais_offset(reference_inter.sites)
        return tuple(
            site.translate(ref_center_bravais_offset - center_bravais_offset) \
            for site in inter.sites
        )
    
    def sort_interaction_tensor_axes(inter, reference_inter):
        translated_sites = get_translated_sites(inter, reference_inter)
        sites_hashes = np.array([hash(site) for site in translated_sites])
        ref_sites_hashes = np.array([hash(site) for site in reference_inter.sites])
        sites_sorting_idxs = sites_hashes.argsort()
        ref_sites_sorting_idxs = ref_sites_hashes.argsort()
        sorted_int_tensor = np.moveaxis(
            inter.interaction_tensor,
            source=sites_sorting_idxs,
            destination=ref_sites_sorting_idxs,
        )

        return sorted_int_tensor

    interactions_by_id = {}

    for inter in interactions:
        # sites_hashes = np.array([hash(site) for site in inter.sites])
        # sites_sorting_idxs = sites_hashes.argsort()
        # sorted_site_hashes = tuple(sites_hashes[sites_sorting_idxs])
        # sorted_int_tensor = inter.interaction_tensor.transpose(sites_sorting_idxs)
        # 
        # if sorted_site_hashes in interactions_by_id:
        #     # NOTE: int_tensors_by_sites[sites] += inter.interaction_tensor
        #     # does not work for some reason.
        #     interactions_by_id[sorted_site_hashes].interaction_tensor \
        #         += sorted_int_tensor
        # else:
        #     sorted_sites = tuple(inter.sites[i] for i in sites_sorting_idxs)
        #     interactions_by_id[sorted_site_hashes] = Interaction(
        #         sites=sorted_sites,
        #         interaction_tensor=inter.interaction_tensor,
        #     )
        inter_id = get_id(inter)
        if inter_id in interactions_by_id:
            reference_inter = interactions_by_id[inter_id]
            sorted_int_tensor = sort_interaction_tensor_axes(inter, reference_inter)
            reference_inter.interaction_tensor += sorted_int_tensor
        else:
            interactions_by_id[inter_id] = inter.copy()

    return list(interactions_by_id.values())







def compute_rotated_interactions(interactions, ground_state_rotation_matrices):
    Rs = ground_state_rotation_matrices
    H = np.array([
        [0.5, 0.5, 0],
        [-0.5j, 0.5j, 0],
        [0, 0, 1]
    ])
    RH = np.einsum("ijk,kl", Rs, H)
    
    rotated_interactions = [
        inter.rotate_spin_coord_system(RH) for inter in interactions
    ]
    return rotated_interactions





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


class BiquadraticHeisenbergInteraction(Interaction):
    def __init__(self, edge: BravaisLattice.Edge, J):
        interaction_tensor = np.outer(np.eye(3).reshape(9), np.eye(3).reshape(9)).reshape((3, 3, 3, 3))
        super().__init__(
            sites=list(edge.get_sites()) * 2,
            interaction_tensor=J * interaction_tensor,
        )