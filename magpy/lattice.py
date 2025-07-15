import numpy as np
from operator import itemgetter
from functools import reduce
from . import util
from typing import List

class BravaisLattice:

    """
    Container class for lattice sites

    bravais_vec: (dim) integer numpy array
    sublattice_index: integer {1..BravaisLattice.num_sites_unit_cell}
    """
    class Site:
        def __init__(self, bravais_coords, sublattice_index):
            self.bravais_coords = np.array(bravais_coords)
            self.subl_idx = int(sublattice_index)

        def translate(self, bravais_coords_delta):
            new_bravais_coords = self.bravais_coords.copy()
            new_bravais_coords += bravais_coords_delta
            return BravaisLattice.Site(
                new_bravais_coords,
                self.subl_idx
            )

        def __eq__(self, other):
            return np.allclose(self.bravais_coords, other.bravais_coords) \
               and self.subl_idx == other.subl_idx

        def __hash__(self):
            return hash(tuple([*self.bravais_coords, self.subl_idx]))
        
        def __str__(self):
            return str((self.bravais_coords, self.subl_idx))
        
        def __repr__(self):
            return f"Site(bravais_coords={self.bravais_coords}, " \
                      + f"subl_idx={self.subl_idx})"

    """
    Container class for edges

    bravais_coords: (dim) integer numpy array
        the Bravais lattice vector connecting the unit cells of the two
        connected lattice sites, expressed in the basis of
        BravaisLattice.bravais_vecs
    sublattice_indices: (2) numpy array
        the sublattice indices of the resp. lattice sites connected by the edge
    """
    class Edge:
        def __init__(self, bravais_coords, sublattice_indices):
            self.bravais_coords = np.array(bravais_coords)
            self.subl_idxs = np.array(sublattice_indices, dtype=np.int64)

        def get_sites(self):
            dim = len(self.bravais_coords)
            return [BravaisLattice.Site(
                bravais_coords=np.zeros(dim, dtype=int),
                sublattice_index=self.subl_idxs[0]
            ), BravaisLattice.Site(
                bravais_coords=self.bravais_coords,
                sublattice_index=self.subl_idxs[1]
            )]
        
        def from_sites(site1, site2):
            return BravaisLattice.Edge(
                site2.bravais_coords - site1.bravais_coords,
                np.array([site1.subl_idx, site2.subl_idx])
            )

        def __eq__(self, other):
            return np.all(self.bravais_coords == other.bravais_coords) \
               and np.all(self.subl_idxs == other.subl_idxs)

        def __hash__(self):
            return hash(tuple([*self.bravais_coords, *self.subl_idxs]))
        
        def __str__(self):
            return str((self.bravais_coords, self.subl_idxs))
        
        def __repr__(self):
            return f"Edge(bravais_coords={self.bravais_coords}, " \
                      + f"subl_idxs={self.subl_idxs})"
        
        def undirected_equals(self, other):
            return self == other \
                or self == self.__class__(
                    -other.bravais_coords, list(reversed(other.subl_idxs)))

    """
    bravais_vecs: (lattice_dim, embedding_dim) float numpy
        The rows are the Bravais lattice vectors,
        lattice_dim is the lattice dimension,
        embedding_dim is the dimension of the embedding space
    num_sites_unit_cell: integer
    edges: set of Edge
    """
    def __init__(self, bravais_vecs, sublattices, edges,
                 reciprocal_high_symmetry_points,
                 open_bc_configs={}):
        self.bravais_vecs = np.array(bravais_vecs)
        self.dim = self.bravais_vecs.shape[0]
        self.embedding_dim = self.bravais_vecs.shape[1]
        if self.embedding_dim < self.dim:
            raise Exception(
                f"Dimension of embedding space {self.embedding_dim} must be " +
                f">= lattice dimension {self.dim}")
        
        self.sublattices = np.array(sublattices)
        if self.sublattices.shape[1] != self.embedding_dim:
            raise Exception(
                f"Dimension of sublattice coordinate vectors " +
                f"{self.sublattices.shape[1]} must be == dimension of " +
                f"embedding space {self.embedding_dim}")

        self.num_sites_unit_cell = len(sublattices)
        self.edges = edges
        self.reciprocal_lattice = ReciprocalLattice(
            bravais_vecs,
            high_symmetry_points=reciprocal_high_symmetry_points)
        self.open_bc_configs = open_bc_configs


    def to_canonical_basis(self, bravais_coords):
        return bravais_coords @ self.bravais_vecs

    def get_canonical_coords_for_site(self, site: Site):
        return self.to_canonical_basis(site.bravais_coords) \
             + self.sublattices[site.subl_idx]
    
    def get_canonical_coords_for_edge(self, edge: Edge):
        return self.to_canonical_basis(edge.bravais_coords) \
             + self.sublattices[edge.subl_idxs[0]] \
             + self.sublattices[edge.subl_idxs[1]]

    def sample_Bravais_lattice_in_Bravais_coords(self, sizes):
        num_unit_cells = int(np.prod(sizes))
        grid_bravais_coords = np.array(np.meshgrid(*[
            np.arange(size) for size in sizes
        ])).reshape((self.dim, num_unit_cells)).T
        
        # shape = (num_unit_cells, self.dim)
        return grid_bravais_coords
    

    def sample_Bravais_lattice_in_canonical_coords(self, sizes):
        num_unit_cells = np.prod(sizes)
        grid_bravais_coords = \
            self.sample_Bravais_lattice_in_Bravais_coords(sizes)
        grid_canonical_coords = np.einsum(
            "xi,in->xn", grid_bravais_coords, self.bravais_vecs)
        
        # shape = (num_unit_cells, self.dim)
        return grid_canonical_coords
    

    def sample_full_lattice_in_canonical_coords(self, sizes):
        num_unit_cells = np.prod(sizes)

        # shape = (num_unit_cells, self.dim)
        grid_canonical_coords = \
            self.sample_Bravais_lattice_in_canonical_coords(sizes)
        
        # shape = (num_unit_cells, len(self.sublattices), self.dim)
        sites_pos = np.zeros((num_unit_cells,
                              self.num_sites_unit_cell,
                              self.sublattices.shape[-1]))
        for x in range(len(grid_canonical_coords)):
            for s in range(len(self.sublattices)):
                sites_pos[x, s] = \
                    grid_canonical_coords[x] + self.sublattices[s]
        sites_pos = sites_pos.reshape(
            (*sizes, len(self.sublattices), self.sublattices.shape[-1]))

        return sites_pos

    """
    sublattice_index: integer {1..num_sites_unit_cell}
        sublattice intex of the site whose neighbor sites should be computed
    n: integer >= 0
        n=0 -> singleton set of the site itself
        n=1 -> nearest neighbors
        n=2 -> next-nearest neighbors
        ...
    
    returns: set of Edge
    """
    def compute_nth_nearest_neighbors(self, sublattice_index, n):
        return self.compute_neighbors(sublattice_index, n_max=n)[-1]
    

    """
    sublattice_index: integer {1..num_sites_unit_cell}
        sublattice intex of the site whose neighbor sites should be computed
    n_max: integer >= 0
        the distance upto which the neighbor sites should be computed
    
    returns: list of set of Edge
        list[0] is the singleton set of the 0-th nearest neighbor site
            (i.e. the site itself)
        list[1] is the set of nearest neighbor sites
        list[2] is the set of next-nearest neighbor sites
        ...
    """
    def compute_neighbors(self, sublattice_index, n_max):
        if n_max == 0:
            return [{
                BravaisLattice.Edge(
                    np.zeros(self.dim, dtype=int), 
                    [sublattice_index]*2
                ),
            }]

        flipped_edges = [BravaisLattice.Edge(
            -edge.bravais_coords,
            list(reversed(edge.subl_idxs))
        ) for edge in self.edges]
        all_edges = self.edges + flipped_edges
        nearest_neighbor_edges = list(filter(
            lambda edge: edge.subl_idxs[0] == sublattice_index,
            all_edges
        ))
        nearest_neighbor_sites = set(map(
            lambda nn_edge: BravaisLattice.Site(
                nn_edge.bravais_coords, nn_edge.subl_idxs[1]),
            nearest_neighbor_edges
        ))

        neighbors_upto_nmax = [set() for _ in range(n_max+1)]
        for nn in nearest_neighbor_sites:
            neighbors_upto_nmax_minus_1_for_nn = \
                self.compute_neighbors(nn.subl_idx, n_max-1)
            nmaxth_nearest_neighbors_for_nn = set(map(
                lambda nm1nn: BravaisLattice.Edge(
                    nm1nn.bravais_coords + nn.bravais_coords,
                    np.array([sublattice_index, nm1nn.subl_idxs[1]])
                ),
                neighbors_upto_nmax_minus_1_for_nn[-1]
            ))

            # filter out duplicate edges
            nmaxth_nearest_neighbors_for_nn = set(filter(
                lambda edge_nmax: all(map(
                    lambda edges_n: edge_nmax not in edges_n,
                    neighbors_upto_nmax_minus_1_for_nn
                )),
                nmaxth_nearest_neighbors_for_nn
            ))

            # add all obtained neighbors for nn to the list neighbors_upto_nmax
            for n in range(n_max):
                neighbors_upto_nmax[n] |= neighbors_upto_nmax_minus_1_for_nn[n]

            neighbors_upto_nmax[n_max] |= nmaxth_nearest_neighbors_for_nn

        return neighbors_upto_nmax

    """
    returns: set of Edge
    """
    def compute_nth_nearest_neighbors_for_entire_unit_cell(self, n):
        nth_nns = set()
        for subl_idx in range(self.num_sites_unit_cell):
            nth_nns_for_subl = self.compute_nth_nearest_neighbors(subl_idx, n)
            # join nth_nns_for_subl with nth_nns, but remove duplicate edges
            # (where 'duplicate' means connecting the same sites but with
            # opposite orientation)
            for nth_nn_for_subl in nth_nns_for_subl:
                if any(map(nth_nn_for_subl.undirected_equals, nth_nns)):
                    continue
                nth_nns.add(nth_nn_for_subl)
        
        return nth_nns



    

    
    
"""
Class describing a reciprocal lattice, providing some helper functions to
assemble momentum paths between high-symmetry points in the Brillouin zone
"""
class ReciprocalLattice:

    class MomentumPath:
        def __init__(self, momenta, point_labels=[], point_coords=[],
                     high_sym_point_idxs=[]):
            if len(point_labels) != len(point_coords) \
            or len(point_labels) != len(high_sym_point_idxs):
                raise Exception("inconsistent high-symmetry point list lengths")
            self.momenta = momenta
            self.ks = momenta
            self.high_sym_point_labels = point_labels
            self.high_sym_point_coords = point_coords
            self.high_sym_point_idxs = high_sym_point_idxs

        def slice(self, k_idxs):
            k_slice = slice(*k_idxs)
            sliced_high_sym_point_idxs = \
                np.array(self.high_sym_point_idxs) - k_idxs[0]
            return ReciprocalLattice.MomentumPath(
                self.ks[k_slice], self.high_sym_point_labels,
                self.high_sym_point_coords, sliced_high_sym_point_idxs)

    """
    bravais_vecs: (lattice_dim, embedding_dim) numpy array
        rows are the Bravais lattice vectors of the corresponding real-space
        lattice
    high_symmetry_points: dict(str -> (lattice_dim) numpy array)
        the high symmetry points in the Brillouin zone, expressed in the basis
        of the reciprocal lattice vectors
    """
    def __init__(self, bravais_vecs, high_symmetry_points):
        self.high_symmetry_points = high_symmetry_points

        # Compute reciprocal lattice vectors with coordinates in the
        # embedding space such that they lie within the lattice plane.
        # This is equivalent to computing the pseudo-inverse of bravais_vecs
        u, s, vh = np.linalg.svd(bravais_vecs, full_matrices=False)
        pseudo_inv = (u * 1/s) @ vh
        self.reciprocal_vecs = 2*np.pi * pseudo_inv \
            if len(bravais_vecs) >= 1 else np.zeros((0, 1))

    """
    Nks: list/array of integer
        how many points to sample in the direction of each reciprocal lattice
        vector
    reciprocal_vecs: numpy array or None
        the vectors forming a parallelepiped from which the momenta are sampled.
        If None, the reciprocal lattice vectors are used
    offset: numpy array or "center" or None

    returns: numpy array
        meshgrid of sampled points
    """
    def sample_inverse_unit_cell(
            self, Nks, reciprocal_vecs=None, offset=None, as_meshgrid=True):
        if len(Nks) != len(self.reciprocal_vecs):
            raise Exception(f"invalid number of axes {len(Nks)}. Should be " \
                          + f"{len(self.reciprocal_vecs)}")
        
        embedding_dim = self.reciprocal_vecs.shape[1]

        # momenta in the basis of the reciprocal lattice vectors
        momenta = np.meshgrid(*[
            np.linspace(0, 1, Nk, endpoint=False) for Nk in Nks
        ])
        # transform to canonical (kx, ky, kz, ...) basis
        if reciprocal_vecs is None:
            reciprocal_vecs = self.reciprocal_vecs
        momenta = np.tensordot(reciprocal_vecs, momenta, axes=[[0], [0]])
        # move momenta by offset
        if offset is None:
            offset = np.zeros(embedding_dim)
        elif type(offset) == str and offset == "center":
            offset = -0.5*np.sum(reciprocal_vecs, axis=0)
        for coord_idx in range(embedding_dim):
            momenta[coord_idx] += offset[coord_idx]

        if not as_meshgrid:
            momenta = momenta.transpose([*range(1, len(Nks)+1), 0])

        return momenta
    

    """
    coords: list of numpy array or dict
        list of reciprocal vectors in the basis of the reciprocal lattice
        vectors

    returns: (list of) numpy array (depending on whether len(list) > 1)
        (list of) reciprocal vectors in the canonical (kx, ky, kz, ...) basis
    """
    def to_canonical_basis(self, coords):
        if type(coords) == dict:
            return dict(
                (k, self.to_canonical_basis([v])) \
                for k, v in coords.items()
            )

        if len(coords) == 1:
            return coords[0] @ self.reciprocal_vecs
        
        return np.array([
            coords @ self.reciprocal_vecs for coords in coords])


    """
    returns: dict
        dict of high-symmetry points in the canonical (kx, ky, kz, ...) basis
    """
    def get_high_symmetry_points_in_canonical_basis(self):
        return self.to_canonical_basis(self.high_symmetry_points)
    
    def _get_dist_fractions(self, point_coords):
        distances = np.array([
            np.linalg.norm(p2 - p1) \
            for p1, p2 in zip(point_coords[1:], point_coords[:-1])
        ])
        total_dist = np.sum(distances)
        dist_fractions = np.array([0, *distances]) / total_dist
        return dist_fractions
    

    def _interpolate_momentum_path_section(self, t, p1, p2, t1, t2):
        t_rel = (t - t1) / (t2 - t1)
        return (1 - t_rel)*p1 + t_rel*p2
    
    def __get_point_labels_and_coords(self, point_labels, custom_hisym_points):
        hisym_points = self.get_high_symmetry_points_in_canonical_basis() \
            if custom_hisym_points is None \
            else self.to_canonical_basis(custom_hisym_points)
        return itemgetter(*point_labels)(hisym_points)


    def get_momentum_path_equally_spaced(self, point_labels, N_k: int,
                                         custom_hisym_points=None):
        if len(point_labels) <= 0:
            raise Exception("should have at least one momentum point.")
        
        point_coords = self.__get_point_labels_and_coords(point_labels,
            custom_hisym_points)
        dist_fractions = self._get_dist_fractions(point_coords)
        cumulative_dist = np.cumsum(dist_fractions)

        momenta = np.zeros((N_k, 2))
        for n, t in enumerate(np.linspace(0, 1, N_k)):
            section_idx = (cumulative_dist > t).argmax() - 1
            momenta[n] = self._interpolate_momentum_path_section(
                t, point_coords[section_idx], point_coords[section_idx+1],
                cumulative_dist[section_idx], cumulative_dist[section_idx+1]
            )
        
        high_sym_point_idxs = list(N_k*cumulative_dist)
        momentum_path = ReciprocalLattice.MomentumPath(momenta, point_labels,
                                                       point_coords,
                                                       high_sym_point_idxs)

        return momentum_path
    

    # length of the returned MomentumPath is not predictable as it depends on
    # the points chosen
    # however, all high-symmetry points lie exactly on the path
    def get_momentum_path_approx_equally_spaced(self, point_labels,
                                                N_ks_for_1st_section,
                                                custom_hisym_points=None):
        point_coords = self.__get_point_labels_and_coords(point_labels,
                                                          custom_hisym_points)
        dist_fractions = self._get_dist_fractions(point_coords)[1:]
        N_ks_for_sections = \
            np.round(N_ks_for_1st_section / dist_fractions[0] * dist_fractions)\
              .astype(int)

        return self.get_momentum_path_custom_spacing(point_labels,
                                                     N_ks_for_sections,
                                                     custom_hisym_points)


    # length of the returned MomentumPath is sum(N_ks_for_section) + 1
    def get_momentum_path_custom_spacing(self, point_labels, N_ks_for_sections,
                                         custom_hisym_points=None):
        point_coords = self.__get_point_labels_and_coords(point_labels,
                                                          custom_hisym_points)

        momenta = np.concatenate(
            (*[np.linspace(point_coords[i], point_coords[i+1],
                        N_ks_for_sections[i], endpoint=False) \
                for i in range(len(N_ks_for_sections))
            ], np.array([point_coords[-1]]))
        )
        
        high_sym_point_idxs = [0] + list(np.cumsum(N_ks_for_sections))
        return ReciprocalLattice.MomentumPath(momenta, point_labels, 
                                              point_coords,
                                              high_sym_point_idxs)













def stack(latt: BravaisLattice, num_layers: int,
          interlayer_edges: List[BravaisLattice.Edge],
          additional_high_symmetry_points=dict(),
          distance_between_layers=1.0,
          sublattice_shifts=None,
          periodic=True,
          additional_bravais_vec=None):
    
    old_dim = latt.dim
    old_embedding_dim = latt.embedding_dim

    if periodic:
        new_dim = latt.dim + 1
        
        if old_dim < old_embedding_dim:  
            # exhaust the extra dimensions of the embedding space
            new_embedding_dim = old_embedding_dim

            if additional_bravais_vec is None:
                # auto-choose a vector orthogonal to the lattice plane
                # as the additional bravais vector along the stacking direction
                _, _, vh = np.linalg.svd(
                    latt.bravais_vecs, full_matrices=True)
                bravais_vecs_ortho_complement = vh[old_dim:]
                additional_bravais_vec = bravais_vecs_ortho_complement[0] \
                    * num_layers * distance_between_layers
                
            new_bravais_vecs = np.r_[
                latt.bravais_vecs,
                additional_bravais_vec,
            ]
        else:
            # extend lattice and embedding space dimension by one
            new_embedding_dim = old_embedding_dim + 1

            if additional_bravais_vec is None:
                # choose (0, ..., 0, 1) as the additional bravais vector along
                # the new dimension
                additional_bravais_vec = np.zeros(new_embedding_dim)
                additional_bravais_vec[-1] = num_layers*distance_between_layers

            new_bravais_vecs = np.r_[
                np.c_[latt.bravais_vecs, np.zeros(old_dim)],
                np.zeros((1, new_dim)),
            ]
            new_bravais_vecs[-1] = additional_bravais_vec
    
        if new_bravais_vecs.shape != (new_dim, new_embedding_dim):
            raise Exception(
                f"{new_dim} new bravais lattice vectors of dimension " +
                f"{new_embedding_dim} are required, but instead, " +
                f"{new_bravais_vecs.shape[0]} vectors of dimension " +
                f"{new_bravais_vecs.shape[1]} have been supplied.")
    else:
        if old_dim < old_embedding_dim:  
            new_bravais_vecs = latt.bravais_vecs
        else:
            new_bravais_vecs = \
                np.c_[latt.bravais_vecs, np.zeros(old_dim)]

    old_num_sites_unit_cell = latt.num_sites_unit_cell
    if periodic:
        pad_subl = \
            lambda subl: subl if old_dim < old_embedding_dim else np.r_[subl, 0]
        normlzd_add_bravais_vec = \
            new_bravais_vecs[-1] / np.linalg.norm(new_bravais_vecs)
        compute_subl_offset = \
            lambda layer: normlzd_add_bravais_vec*layer*distance_between_layers
        new_sublattices = np.array([
            pad_subl(subl) + compute_subl_offset(layer) \
            for layer in range(num_layers) \
            for subl in latt.sublattices
        ])
    else:
        new_sublattices = np.array([
            np.r_[subl, layer*distance_between_layers] \
            for layer in range(num_layers) \
            for subl in latt.sublattices
        ])
    if sublattice_shifts is not None:
        new_sublattices += sublattice_shifts

    new_edges = []
    for edge in latt.edges:
        new_edges_for_old_edge = [
            BravaisLattice.Edge(
                np.r_[edge.bravais_coords, 0] \
                    if periodic else edge.bravais_coords,
                edge.subl_idxs + old_num_sites_unit_cell*layer
            ) \
            for layer in range(num_layers)
        ]
        new_edges += new_edges_for_old_edge
    new_edges += interlayer_edges

    old_high_sym_points = latt.reciprocal_lattice.high_symmetry_points
    if periodic:
        new_high_sym_points = dict(
            (k, np.r_[v, 0]) \
            for k, v in old_high_sym_points.items()
        )
    else:
        new_high_sym_points = dict(**old_high_sym_points)

    new_high_sym_points.update(additional_high_symmetry_points)

    stacked_lattice = BravaisLattice(
        new_bravais_vecs, new_sublattices, new_edges,
        new_high_sym_points
    )

    return stacked_lattice



"""
This class is only for internal use to avoid code duplication in
lattice.transform and models.transform
"""
class __TransformUtils:
    def __init__(self, bravais_trans, old_num_sites_unit_cell):
        self.trans_active = bravais_trans.T
        self.trans_passive = np.linalg.inv(self.trans_active)
        self.dim = bravais_trans.shape[0]
        self.old_num_sites_unit_cell = old_num_sites_unit_cell
        self.EPS = 1e-12 # to account for floating point inaccuarcies

        self.new_unit_cell_sites_coords = self.get_new_unit_cell_sites_coords()


    def get_new_unit_cell_sites_coords(self):
        pts = np.stack(([0, 1],)*self.dim, 0)
        corners_of_parallelepiped_in_new_coords = \
            (np.array(np.meshgrid(*pts)).T).reshape((2**self.dim, self.dim)).T
        corners_of_parallelepiped_in_old_coords = \
            self.trans_active @ corners_of_parallelepiped_in_new_coords
        
        # in coordinates of the old bravais lattice vectors
        new_unit_cell_sites_coords = np.array(np.meshgrid(*[
            np.arange(
                np.amin(corners_of_parallelepiped_in_old_coords[d]),
                np.amax(corners_of_parallelepiped_in_old_coords[d]) + 1
            ) \
            for d in range(self.dim)
        ])).astype(int)
        new_unit_cell_sites_coords = new_unit_cell_sites_coords.reshape(
            (self.dim, np.prod(new_unit_cell_sites_coords.shape[1:]))).T
        
        # filter out all sites that are not inside the parallelogram
        # (parallelepiped) spanned by the new bravais vectors.
        # Note: we need self.EPS here to account for floating point issues, 
        # e.g. coordinates being e.g. 0.99999999999999999 instead of 1,
        # or -0.0000000000000001 instead of 0, etc.
        new_unit_cell_sites_coords = list(filter(
            lambda site_in_new_coords: 
                np.all(self.trans_passive @ site_in_new_coords >= 0 - self.EPS) and \
                np.all(self.trans_passive @ site_in_new_coords < 1 - self.EPS),
            new_unit_cell_sites_coords
        ))

        return new_unit_cell_sites_coords

    def map_site_to_enlarged_coord_system(self, site, new_unit_cell_sites_coords):
        # Note: we need self.EPS here to account for floating point issues, 
        # e.g. coordinates being e.g. 0.99999999999999999 instead of 1,
        # or -0.0000000000000001 instead of 0, etc.
        new_coords = self.trans_passive @ site.bravais_coords
        new_bravais_coords = (new_coords + self.EPS) // 1
        new_coords_remainder = self.trans_active @ (new_coords - new_bravais_coords)

        idx = -1
        for n, coords in enumerate(new_unit_cell_sites_coords):
            if np.allclose(coords, new_coords_remainder):
                idx = n
                break

        if idx < 0:
            raise Exception(f"Internal error: negative index {idx}. J.H. needs to debug harder!")

        new_subl_idx = site.subl_idx + self.old_num_sites_unit_cell * idx
        return BravaisLattice.Site(
            new_bravais_coords,
            new_subl_idx
        )
    
    def get_all_translated_sites(self, old_site, new_unit_cell_sites_coords):
        return [
            old_site.translate(site_delta) \
            for site_delta in new_unit_cell_sites_coords
        ]
    
    def get_all_translated_sites_in_enlarged_coord_system(
            self, old_site, new_unit_cell_sites_coords):
        sites_in_enlarged_coord_sys = [
            self.map_site_to_enlarged_coord_system(site, new_unit_cell_sites_coords) \
            for site in self.get_all_translated_sites(old_site, new_unit_cell_sites_coords)
        ]
        return sites_in_enlarged_coord_sys
    
    def get_all_translated_edges_in_enlarged_coord_system(
            self, old_edge, new_unit_cell_sites_coords):
        old_site1, old_site2 = old_edge.get_sites()
        new_sites1 = self.get_all_translated_sites_in_enlarged_coord_system(
            old_site1, new_unit_cell_sites_coords)
        new_sites2 = self.get_all_translated_sites_in_enlarged_coord_system(
            old_site2, new_unit_cell_sites_coords)
        return [
            BravaisLattice.Edge(
                site2.bravais_coords - site1.bravais_coords,
                np.array([site1.subl_idx, site2.subl_idx])
            ) \
            for site1, site2 in zip(new_sites1, new_sites2)
        ]



"""
The last parameter __return_with_transform_utils is only for internal use,
to avoid recalculating some quantities in models.transform method
"""
def transform(latt: BravaisLattice, bravais_trans, 
              __return_with_transform_utils=False):
    
    dim = bravais_trans.shape
    if len(dim) != 2 or dim[0] != latt.dim or dim[0] != dim[1]:
        raise Exception(f"lattice dimension {latt.dim} " +
                        f" and bravais_trans dimensions {dim} should be " +
                        f"the same.")

    old_lattice = latt

    transform_utils = __TransformUtils(
        bravais_trans, old_lattice.num_sites_unit_cell,
    )

    trans_active = transform_utils.trans_active
    new_bravais_vecs = trans_active.T @ old_lattice.bravais_vecs

    new_sublattices = np.array([
        old_lattice.bravais_vecs.T @ coord + subl \
        for coord in transform_utils.new_unit_cell_sites_coords \
        for subl in old_lattice.sublattices
    ])
    
    new_edges = []
    for old_edge in old_lattice.edges:
        new_edges += \
            transform_utils.get_all_translated_edges_in_enlarged_coord_system(
                old_edge, transform_utils.new_unit_cell_sites_coords)

    new_hisym_points = dict(
        (k, trans_active.T @ v) \
        for k, v in old_lattice.reciprocal_lattice.high_symmetry_points.items()
    )

    lattice_with_enlarged_unit_cell = BravaisLattice(
        new_bravais_vecs,
        new_sublattices,
        new_edges,
        new_hisym_points
    )


    if __return_with_transform_utils:
        return lattice_with_enlarged_unit_cell, transform_utils
    
    return lattice_with_enlarged_unit_cell





"""
This class is only for internal use to avoid code duplication in
lattice.delete_dimensions and models.delete_dimensions
"""
class __DeleteDimensionsUtils:
    @staticmethod
    def filter(objects, get_bravais_coords, dims, periodic_boundary_conditions):
        for obj in objects:
            bravais_coords = get_bravais_coords(obj) 
            bravais_coords_lower_dim = np.delete(bravais_coords, dims, axis=-1)
            if periodic_boundary_conditions \
            or np.all(np.take(bravais_coords, dims, axis=-1) == 0):
                yield obj, bravais_coords_lower_dim


def delete_dimensions(latt: BravaisLattice, dims, new_bravais_vecs,
                      new_high_symmetry_points=None,
                      periodic_boundary_conditions=False):

    if new_bravais_vecs.shape[0] != latt.dim - len(dims):
        raise Exception(f"number supplied new bravais vectors " + 
                        f"{new_bravais_vecs.shape[0]} must equal the " +
                        f"lattice dimension of the new lower-dimensional " +
                        f"lattice {latt.dim - len(dims)}")
    
    if new_bravais_vecs.shape[1] != latt.embedding_dim:
        raise Exception(f"dimension of supplied new bravais vectors " + 
                        f"{new_bravais_vecs.shape[1]} must equal the " +
                        f"embedding dimension of the lattice " +
                        f"{latt.embedding_dim}")

    new_edges = [
        BravaisLattice.Edge(
            filtered_bravais_coords, edge.subl_idxs.copy()
        ) \
        for edge, filtered_bravais_coords in __DeleteDimensionsUtils.filter(
            latt.edges,
            lambda edge: edge.bravais_coords,
            dims, periodic_boundary_conditions,
        )
    ]

    if new_high_symmetry_points is None:
        new_high_symmetry_points = dict(
            (k, np.delete(v, dims)) \
            for k, v in latt.reciprocal_lattice. \
                        high_symmetry_points.items()
        )

    new_lattice = BravaisLattice(
        new_bravais_vecs, latt.sublattices.copy(),
        new_edges, new_high_symmetry_points
    )

    return new_lattice






def rearrange_sublattices(latt: BravaisLattice, permutation):
    new_sublattices = util.permute(latt.sublattices, permutation)

    new_edges = [
        BravaisLattice.Edge(
            bravais_coords=edge.bravais_coords,
            sublattice_indices=[
                permutation[subl_idx] for subl_idx in edge.subl_idxs
            ],
        ) for edge in latt.edges
    ]

    new_high_symmetry_points = {
        k: v.copy() \
        for k, v in latt.reciprocal_lattice.high_symmetry_points.items()
    }

    new_lattice = BravaisLattice(
        latt.bravais_vecs.copy(), new_sublattices,
        new_edges, new_high_symmetry_points,
    )

    return new_lattice








class SimpleCubicLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            sublattices=np.array([[0, 0, 0]]),
            edges=[
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([0, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0, 0]),
                "X": np.array([1/2, 0, 0]),
                "M": np.array([1/2, 1/2, 0]),
                "R": np.array([1/2, 1/2, 1/2]),
            },
            open_bc_configs={} # TODO
        )



class BodyCenteredCubicLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            sublattices=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            edges=[
                # edges along cubic axes
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([0, 0])),
                # edges along space diagonals connecting to the body center
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 1])),
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([1, 1, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([1, 0, 1]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 1, 1]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([1, 1, 1]), np.array([1, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0, 0]),
                "H": np.array([0, 0, 1]),
                "N": np.array([0, 1/2, 1/2]),
                "P": np.array([1/2, 1/2, 1/2]),
            },
            open_bc_configs={} # TODO
        )


class BodyCenteredCubicPrimitiveLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
            ]),
            sublattices=np.array([[0, 0, 0]]),
            edges=[
                # just nearest neighbor edges along the space diagonals
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 0, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 1, 1]), np.array([0, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0, 0]),
                "H": np.array([-1/2, 1/2, 1/2]),
                "N": np.array([0, 1/2, 0]),
                "P": np.array([1/4, 1/4, 1/4]),
            },
            open_bc_configs={} # TODO
        )


class FaceCenteredCubicLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            sublattices=np.array([
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ]),
            edges=[
                # edges along cubic axes
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([0, 0])),
                # edges along face diagonals connecting to the face centers
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 1])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 2])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 3])),

                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([1, 1, 0]), np.array([1, 0])),

                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([2, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([2, 0])),
                BravaisLattice.Edge(np.array([1, 0, 1]), np.array([2, 0])),

                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([3, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([3, 0])),
                BravaisLattice.Edge(np.array([0, 1, 1]), np.array([3, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0, 0]),
                "X": np.array([0, 1, 0]),
                "L": np.array([1/2, 1/2, 1/2]),
                "W": np.array([1/2, 1, 0]),
                "U": np.array([1/4, 1, 1/4]),
                "K": np.array([3/2, 3/2, 0]),
            },
            open_bc_configs={} # TODO
        )


class FaceCenteredCubicPrimitiveLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                [0, 0.5, 0.5],
            ]),
            sublattices=np.array([[0, 0, 0]]),
            edges=[
                # just nearest neighbor edges along the face diagonals
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 1, 0]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 0, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([0, 1, 1]), np.array([0, 0])),
                BravaisLattice.Edge(np.array([1, 1, 1]), np.array([0, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0, 0]),
                "X": np.array([0, 1/2, 1/2]),
                "L": np.array([1/2, 1/2, 1/2]),
                "W": np.array([1/4, 3/4, 1/2]),
                "U": np.array([1/4, 5/8, 5/8]),
                "K": np.array([3/8, 3/4, 3/8]),
            },
            open_bc_configs={} # TODO
        )


class DiamondLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                [0, 0.5, 0.5],
            ]),
            sublattices=np.array([[0, 0, 0], [0.25, 0.25, 0.25]]),
            edges=[
                # intra-unit-cell edge
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 1])),
                # inter-unit-cell edges
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([1, 0])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([1, 0])),
            ],
            reciprocal_high_symmetry_points=\
                FaceCenteredCubicPrimitiveLattice(). \
                reciprocal_lattice.high_symmetry_points,
            open_bc_configs={} # TODO
        )


class PyrochloreLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                [0, 0.5, 0.5],
            ]),
            sublattices=np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 1, 1],
            ]) / 4.0,
            edges=[
                # intra-unit-cell tetrahedron edges
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 1])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 2])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([0, 3])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([1, 2])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([1, 3])),
                BravaisLattice.Edge(np.array([0, 0, 0]), np.array([2, 3])),
                # inter-unit-cell edges
                BravaisLattice.Edge(np.array([1, 0, 0]), np.array([3, 0])),
                BravaisLattice.Edge(np.array([0, 1, 0]), np.array([3, 1])),
                BravaisLattice.Edge(np.array([0, 0, 1]), np.array([3, 2])),
                BravaisLattice.Edge(np.array([0, 1, -1]), np.array([2, 1])),
                BravaisLattice.Edge(np.array([1, 0, -1]), np.array([2, 0])),
                BravaisLattice.Edge(np.array([1, -1, 0]), np.array([1, 0])),
            ],
            reciprocal_high_symmetry_points=\
                FaceCenteredCubicPrimitiveLattice(). \
                reciprocal_lattice.high_symmetry_points,
            open_bc_configs={} # TODO
        )



class HoneycombLatticeA(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [-np.sqrt(3)/2, 3/2],
                [np.sqrt(3)/2, 3/2],
            ]),
            sublattices=np.array([
                [0, 0],
                [0, -1]
            ]),
            edges=[
                #vertical bond
                BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                #lowerright-upperleft 60° bond
                BravaisLattice.Edge(np.array([1, 0]), np.array([0, 1])),
                #lowerleft-upperright 60° bond
                BravaisLattice.Edge(np.array([0, 1]), np.array([0, 1])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0]),
                "M": np.array([0, 1/2]),
                "K": np.array([1/3, 2/3]),
                "K'": np.array([-1/3, 2/3]),
                "X": np.array([1/2, -1/2]),
                "Y": np.array([1/2, 1/2]),
            },
            open_bc_configs={
                "zigzag": {
                    "surface": np.array([[1, 0]]),
                    "normal": np.array([[0, 1]]),
                },
                "dangling": {
                    "surface": np.array([[-1, 1]]),
                    "normal": np.array([[1, 0]]),
                },
                "armchair": {
                    "surface": np.array([[1, 1]]),
                    "normal": np.array([[0, 1]]),
                },
            }
        )


class HoneycombLatticeB(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [3/2, np.sqrt(3)/2],
                [-3/2, np.sqrt(3)/2],
            ]),
            sublattices=np.array([
                [0, 0],
                [1, 0]
            ]),
            edges=[
                #horizontal bond
                BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                #upperleft-lowerright 60° bond
                BravaisLattice.Edge(np.array([1, 0]), np.array([0, 1])),
                #upperright-lowerleft 60° bond
                BravaisLattice.Edge(np.array([0, 1]), np.array([1, 0])),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0]),
                "Gamma'1": np.array([1, 0]),
                "Gamma'2": np.array([1, -1]),
                "M": np.array([0, 1/2]),
                "K": np.array([-1/3, 2/3]),
                "K'": np.array([-2/3, 1/3]),
                "X": np.array([1/2, -1/2]),
                "Y": np.array([1/2, 1/2]),
            },
            open_bc_configs={
                "zigzag": {
                    "surface": np.array([[1, 0]]),
                    "normal": np.array([[0, 1]]),
                },
                "dangling": {
                    "surface": np.array([[1, 1]]),
                    "normal": np.array([[0, 1]]),
                },
                "armchair": {
                    "surface": np.array([[-1, 1]]),
                    "normal": np.array([[0, 1]]),
                },
            }
        )



class SquareLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [1, 0],
                [0, 1]
            ]),
            sublattices=np.array([
                [0, 0]
            ]),
            edges=[
                #horizontal bond
                BravaisLattice.Edge(np.array([1, 0]), np.zeros(2, dtype=int)),
                #vertical bond
                BravaisLattice.Edge(np.array([0, 1]), np.zeros(2, dtype=int)),
            ],
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0, 0]),
                "M": np.array([0.5, 0.5]),
                "X": np.array([0.5, 0]),
                "Y": np.array([0, 0.5]),
            },
            open_bc_configs={
                "horizontal": {
                    "surface": np.array([[1, 0]]),
                    "normal": np.array([[0, 1]]),
                },
                "vertical": {
                    "surface": np.array([[0, 1]]),
                    "normal": np.array([[1, 0]]),
                },
                "both": {
                    "surface": np.zeros((0, 2), dtype=int),
                    "normal": np.array([[0, 1], [1, 0]]),
                },
            }
        )



class KagomeLattice(BravaisLattice):
    def __init__(self):
        super().__init__(
            bravais_vecs=np.array([
                [-1, np.sqrt(3)],
                [1, np.sqrt(3)]
            ]),
            sublattices=np.array([
                [1/2, np.sqrt(3)/2], [0, 0], [1, 0]
            ]),
            edges=[
                #intra-unit-cell triangle
                BravaisLattice.Edge(np.array([0, 0]), np.array([0, 1])),
                BravaisLattice.Edge(np.array([0, 0]), np.array([1, 2])),
                BravaisLattice.Edge(np.array([0, 0]), np.array([2, 0])),
                #horizontal inter-unit-cell bond
                BravaisLattice.Edge(np.array([-1, 1]), np.array([2, 1])),
                #upperleft-lowerright 60° bond
                BravaisLattice.Edge(np.array([1, 0]), np.array([0, 2])),
                #upperright-lowerleft 60° bond
                BravaisLattice.Edge(np.array([0, 1]), np.array([0, 1])),
            ],
            reciprocal_high_symmetry_points={   # following Chernyshev
                "Gamma": np.array([0, 0]),
                "M": np.array([0, 1/2]),
                "K": np.array([-1/3, 1/3]),
            },
            open_bc_configs={
                "clean": {
                    "surface": np.array([[-1, 1]]),
                    "normal": np.array([[0, 1]]),
                },
                "alternating": {
                    "surface": np.array([[1, 1]]),
                    "normal": np.array([[0, 1]]),
                },
            }
        )


"""
1D chain
"""
class ChainLattice(BravaisLattice):
    def __init__(self, num_sites_unit_cell=1, edges=None):
        if edges == None:
            edges = [
                BravaisLattice.Edge(np.array([0]), np.array([n, n+1])) \
                for n in range(num_sites_unit_cell-1)
            ] + [
                BravaisLattice.Edge(np.array([1]),
                                    np.array([num_sites_unit_cell-1, 0]))
            ]

        super().__init__(
            bravais_vecs=np.array([
                [1.0]
            ]),
            sublattices=np.array([
                np.linspace(0, 1, num_sites_unit_cell, endpoint=False)
            ]).T,
            edges=edges,
            reciprocal_high_symmetry_points={
                "Gamma": np.array([0]),
                "Gamma'": np.array([1]),
                "M": np.array([1/2]),
            }
        )


"""
0D dot
"""
class DotLattice(BravaisLattice):
    def __init__(self, num_sites_unit_cell=1, edges=None):
        if edges is None:
            edges = [
                BravaisLattice.Edge(np.array([]), np.array([n, n+1])) \
                for n in range(num_sites_unit_cell-1)
            ]

        super().__init__(
            bravais_vecs=np.zeros((0, 1)),
            sublattices=np.array([
                np.linspace(0, 1, num_sites_unit_cell)
            ], dtype=int).T,
            edges=edges,
            reciprocal_high_symmetry_points={
                "Gamma": np.array([]),
            }
        )


