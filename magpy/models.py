import numpy as np
from .lattice import BravaisLattice, ReciprocalLattice, DotLattice
from .interactions import Interaction, CompositeInteraction
from . import util
from typing import List

class Model:

    """
    lattice: BravaisLattice
    interactions: list of Interaction
    classical_ground_state: (lattice.num_sites_unit_cell, 3) numpy array
        defines the ground state orientation of the spins in the semi-classical
        limit. Their lengths specify the on-site spin quantum number S
    """
    def __init__(self, lattice: BravaisLattice, interactions,
                 classical_ground_state):
        classical_gs_required_shape = (lattice.num_sites_unit_cell, 3)
        if classical_ground_state.shape != classical_gs_required_shape:
            raise Exception("classical ground state array has wrong shape " \
                + f"{classical_ground_state.shape}. Should be " \
                + f"{classical_gs_required_shape}")
        self.lattice = lattice
        self.interactions = CompositeInteraction(interactions).expand()
        self.classical_gs = classical_ground_state

    """
    The on-site spin quantum number S is characterized by the length of the
    corresponding classical ground state vector.
    """
    def get_onsite_spin_quantum_numbers(self):
        return np.linalg.norm(self.classical_gs, axis=1)

    """
    For each site i in the unit cell, compute an orthogonal matrix R such that
        R @ [0, 0, 1].T = self.classical_gs[i].T
    so that, for the spin operators, we get
        S = R @ S'
    where S (S') is the spin operator in the "natural" (rotated HP) basis.
    """
    def compute_ground_state_rotation_matrices(self):
        Rs = np.zeros((self.lattice.num_sites_unit_cell, 3, 3))
        for R, gs_for_site in zip(Rs, self.classical_gs):
            R[:, 2] = gs_for_site
            # choose a vector orthogonal to R[:, 2] (this is a gauge choice!)
            R[:, 1] = np.array([0, 1, 0]) \
                if np.allclose(R[0:2, 2], np.zeros(2)) \
                else np.array([-R[1, 2], R[0, 2], 0])
            # compute a vector which forms a right-handed triad with R[:,(1,2)]
            R[:, 0] = np.cross(R[:, 1], R[:, 2])

        # normalize the columns
        Rs /= np.linalg.norm(Rs, axis=1, keepdims=True)

        return Rs

    """
    Rotate each spin from the "natural" basis (S) to the HP basis (S'), and then
    transform (S'^x, S'^y) to (S'^+, S'^-) by a Hadamard gate:
        S = R @ S' = R @ H @ [S'^+, S'^-, S'^z].T
    An interaction like
        T_{ij...} S_i S_j ...
    transforms like
        T_{ij...} RH_{ii'} RH_{jj'} ... S'_{i'} S'_{j'} ...
    so the transformed interaction is
        T'_{ij...} = T_{ij...} RH_{ii'} RH_{jj'} ...
    """
    def compute_rotated_interactions(self):
        Rs = self.compute_ground_state_rotation_matrices()
        H = np.array([
            [0.5, 0.5, 0],
            [-0.5j, 0.5j, 0],
            [0, 0, 1]
        ])
        RH = np.einsum("ijk,kl", Rs, H)
        
        rotated_interactions = [
            inter.rotate_spin_coord_system(RH) for inter in self.interactions
        ]
        return rotated_interactions



"""
Create num_layers copies of the model and link the layers by interlayer edges
and interlayer interactions. This increases the unit cell size by a factor of
num_layers. If periodic == True, the lattice dimension is increased by one
and periodic boundary conditions are imposed between the first and the last
copy.
Irrespective of the value of periodic, if the lattice dimension equals the
embedding dimension, the embedding dimension is also increased by one
(so that the embedding space can accommodate the extended sublattice positions,
higher-dimensional lattice).

distance_between_layers: float
    the physical distance between the individual layers
sublattice_shifts: None or numpy array
    a total of N vectors (where N is the unit cell size of the extended model,
    i.e. the unit cell size of the original model times the number of layers)
    which have the dimension of the embedding space of the extended model.
    Specifies additional relative shifts of the sublattices of the extended
    model.
additional_bravais_vec: None or numpy array
    only considered if periodic == True; specifies an explicit additional
    lattice vector along the new periodic direction. If omitted, the function
    will use either (0, ..., 0, 1) [if the lattice dimension equals the
    embedding dimension], or an arbitrary orthogonal vector to the lattice plane
    [if the lattice dimension is smaller than the embedding dimension].
    The layers in the new periodic direction will be offset along
    additional_bravais_vec.
"""
def stack(model: Model, num_layers: int,
          interlayer_edges: List[BravaisLattice.Edge],
          interlayer_interactions: List[Interaction],
          additional_high_symmetry_points=dict(),
          new_classical_ground_state=None,
          distance_between_layers=1.0,
          sublattice_shifts=None,
          periodic=True,
          additional_bravais_vec=None):
    
    old_dim = model.lattice.dim
    old_embedding_dim = model.lattice.embedding_dim

    if periodic:
        new_dim = model.lattice.dim + 1
        
        if old_dim < old_embedding_dim:  
            # exhaust the extra dimensions of the embedding space
            new_embedding_dim = old_embedding_dim

            if additional_bravais_vec is None:
                # auto-choose a vector orthogonal to the lattice plane
                # as the additional bravais vector along 
                _, _, vh = np.linalg.svd(
                    model.lattice.bravais_vecs, full_matrices=True)
                bravais_vecs_ortho_complement = vh[old_dim:]
                additional_bravais_vec = bravais_vecs_ortho_complement[0] \
                    * num_layers * distance_between_layers
                
            new_bravais_vecs = np.r_[
                model.lattice.bravais_vecs,
                additional_bravais_vec,
            ]
        else:
            # extend lattice and embedding space dimension by one and
            # choose (0, ..., 0, 1) as the additional bravais vector along
            # the new dimension
            new_embedding_dim = old_embedding_dim + 1

            new_bravais_vecs = np.r_[
                np.c_[model.lattice.bravais_vecs, np.zeros(old_dim)],
                np.zeros((1, new_dim)),
            ]
            new_bravais_vecs[-1, -1] = num_layers * distance_between_layers
    
        if new_bravais_vecs.shape != (new_dim, new_embedding_dim):
            raise Exception(
                f"{new_dim} new bravais lattice vectors of dimension " +
                f"{new_embedding_dim} are required, but instead, " +
                f"{new_bravais_vecs.shape[0]} vectors of dimension " +
                f"{new_bravais_vecs.shape[1]} have been supplied.")
    else:
        if old_dim < old_embedding_dim:  
            new_bravais_vecs = model.lattice.bravais_vecs
        else:
            new_bravais_vecs = \
                np.c_[model.lattice.bravais_vecs, np.zeros(old_dim)]

    old_num_site_unit_cell = model.lattice.num_sites_unit_cell
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
            for subl in model.lattice.sublattices
        ])
    else:
        new_sublattices = np.array([
            np.r_[subl, layer*distance_between_layers] \
            for layer in range(num_layers) \
            for subl in model.lattice.sublattices
        ])
    if sublattice_shifts is not None:
        new_sublattices += sublattice_shifts

    new_edges = []
    for edge in model.lattice.edges:
        new_edges_for_old_edge = [
            BravaisLattice.Edge(
                np.r_[edge.bravais_coords, 0] \
                    if periodic else edge.bravais_coords,
                edge.subl_idxs + old_num_site_unit_cell*layer
            ) \
            for layer in range(num_layers)
        ]
        new_edges += new_edges_for_old_edge
    new_edges += interlayer_edges

    old_high_sym_points = model.lattice.reciprocal_lattice.high_symmetry_points
    if periodic:
        new_high_sym_points = dict(
            (k, np.r_[v, 0]) \
            for k, v in old_high_sym_points.items()
        )
    else:
        new_high_sym_points = dict(**old_high_sym_points)

    new_high_sym_points.update(additional_high_symmetry_points)

    new_lattice = BravaisLattice(
        new_bravais_vecs, new_sublattices, new_edges,
        new_high_sym_points
    )



    new_interactions = []
    for inter in model.interactions:
        new_inters = [
            Interaction([
                BravaisLattice.Site(
                    np.r_[site.bravais_coords, 0] \
                        if periodic else site.bravais_coords,
                    site.subl_idx + old_num_site_unit_cell*layer
                ) for site in inter.sites
            ], inter.interaction_tensor.copy()) \
            for layer in range(num_layers)
        ]
        new_interactions += new_inters
    new_interactions += interlayer_interactions

    if new_classical_ground_state == None:
        new_classical_ground_state = np.tile(
            model.classical_gs, (num_layers, 1)
        )

    extended_model = Model(
        new_lattice, new_interactions, new_classical_ground_state
    )



    return extended_model


"""
Form the "direct product" of model1 and model2.
model1 and model2 must have the same underlying Bravais lattice.
inter_model_interactions specifies additional "off-diagonal" interactions
between sites of model1 and sites of model2.
"""
def direct_product(model1: Model, model2: Model,
          inter_model_interactions=[],
          sublattice_shifts=None,
          new_classical_ground_state=None):
    old_lattice1, old_lattice2 = model1.lattice, model2.lattice
    if old_lattice1.dim != old_lattice2.dim:
        raise Exception("dimensions do not match: " + \
            f"{old_lattice1.dim} and {old_lattice2.dim}.")
    
    if not np.allclose(old_lattice1.bravais_vecs, old_lattice2.bravais_vecs):
        raise Exception("Bravais lattice vectors do not match: " + \
            f"{old_lattice1.bravais_vecs} and {old_lattice2.bravais_vecs}.")
    
    duplicate_hisym_points = dict(
        (label1, (hisym_point1, hisym_point2)) \
        for label1, hisym_point1 in \
            old_lattice1.reciprocal_lattice.high_symmetry_points.items() \
        for label2, hisym_point2 in \
            old_lattice2.reciprocal_lattice.high_symmetry_points.items() \
        if label1 == label2 and np.any(hisym_point1 != hisym_point2)
    )
    if len(duplicate_hisym_points) > 0:        
        raise Exception("High-symmetry points do not match: " + \
            f"{duplicate_hisym_points}.")

    new_hisym_points = dict(
        **old_lattice1.reciprocal_lattice.high_symmetry_points)
    new_hisym_points.update(
        old_lattice2.reciprocal_lattice.high_symmetry_points)
    
    new_sublattices = np.concatenate(
        (old_lattice1.sublattices, old_lattice2.sublattices))
    if sublattice_shifts is not None:
        new_sublattices += sublattice_shifts

    new_lattice = BravaisLattice(
        old_lattice1.bravais_vecs.copy(),
        new_sublattices,
        old_lattice1.edges + list(map(
            lambda edge: BravaisLattice.Edge(
                edge.bravais_coords.copy(),
                edge.subl_idxs + old_lattice1.num_sites_unit_cell
            ),
            old_lattice2.edges)),
        new_hisym_points
    )


    new_interactions = model1.interactions
    new_interactions += list(map(
        lambda inter: Interaction(list(map(
            lambda site: BravaisLattice.Site(
                site.bravais_coords.copy(),
                site.subl_idx + old_lattice1.num_sites_unit_cell
            ), inter.sites
        )), inter.interaction_tensor.copy()),
        model2.interactions
    ))
    new_interactions += inter_model_interactions

    if new_classical_ground_state is None:
        new_classical_ground_state = \
            np.concatenate((model1.classical_gs, model2.classical_gs))

    stacked_model = Model(
        new_lattice,
        new_interactions,
        new_classical_ground_state)



    return stacked_model


"""
Transform the bravais lattice of model to a new basis, potentially enlarging the
unit cell by an integer factor equal to the determinant of the basis transform
matrix.
bravais_trans is a quadratic array of integers.
Its rows are the new bravais vectors expressed in coordinates of the old ones.
"""
def transform(model: Model, bravais_trans):
    dim = bravais_trans.shape
    if len(dim) != 2 or dim[0] != model.lattice.dim or dim[0] != dim[1]:
        raise Exception(f"lattice dimension {model.lattice.dim} " +
                        f" and bravais_trans dimensions {dim} should be " +
                        f"the same.")

    old_lattice = model.lattice

    trans_active = bravais_trans.T
    trans_passive = np.linalg.inv(trans_active)
    new_bravais_vecs = trans_active.T @ old_lattice.bravais_vecs


    pts = np.stack(([0, 1],)*dim[0],0)
    corners_of_parallelepiped_in_new_coords = \
        (np.array(np.meshgrid(*pts)).T).reshape((2**dim[0], dim[0])).T
    corners_of_parallelepiped_in_old_coords = \
        trans_active @ corners_of_parallelepiped_in_new_coords
    # in coordinates of the old bravais lattice vectors
    new_unit_cell_sites_coords = np.array(np.meshgrid(*[
        np.arange(
            np.amin(corners_of_parallelepiped_in_old_coords[d]),
            np.amax(corners_of_parallelepiped_in_old_coords[d]) + 1
        ) \
        for d in range(dim[0])
    ])).astype(int)
    new_unit_cell_sites_coords = new_unit_cell_sites_coords.reshape(
        (dim[0], np.prod(new_unit_cell_sites_coords.shape[1:]))).T
    
    # filter out all sites that are not inside the parallelogram
    # (parallelepiped) spanned by the new bravais vectors
    new_unit_cell_sites_coords = list(filter(
        lambda site_in_new_coords: 
            np.all(trans_passive @ site_in_new_coords >= 0) and \
            np.all(trans_passive @ site_in_new_coords < 1),
        new_unit_cell_sites_coords
    ))

    old_num_sites_unit_cell = old_lattice.num_sites_unit_cell
    scale_factor = int(np.round(np.abs(np.linalg.det(trans_active))))
    # new_num_sites_unit_cell = old_num_sites_unit_cell * scale_factor
    new_sublattices = np.array([
        model.lattice.bravais_vecs.T @ coord + subl \
        for coord in new_unit_cell_sites_coords \
        for subl in model.lattice.sublattices
    ])
    
    def map_site_to_enlarged_coord_system(site):
        new_coords = trans_passive @ site.bravais_coords
        new_bravais_coords = new_coords // 1
        new_coords_remainder = trans_active @ (new_coords % 1)

        idx = -1
        for n, coords in enumerate(new_unit_cell_sites_coords):
            if np.allclose(coords, new_coords_remainder):
                idx = n
                break

        new_subl_idx = site.subl_idx + old_num_sites_unit_cell * idx
        return BravaisLattice.Site(
            new_bravais_coords,
            new_subl_idx
        )

    def translate_site(site, delta):
        new_bravais_coords = site.bravais_coords.copy()
        new_bravais_coords += delta
        return BravaisLattice.Site(
            new_bravais_coords,
            site.subl_idx
        )
    
    def get_all_translated_sites(old_site):
        return [
            translate_site(old_site, site_delta) \
            for site_delta in new_unit_cell_sites_coords
        ]
    
    def get_all_translated_sites_in_enlarged_coord_system(old_site):
        sites_in_enlarged_coord_sys = [
            map_site_to_enlarged_coord_system(site) \
            for site in get_all_translated_sites(old_site)
        ]
        return sites_in_enlarged_coord_sys
    
    def get_all_translated_edges_in_enlarged_coord_system(old_edge):
        old_site1, old_site2 = old_edge.get_sites()
        new_sites1 = get_all_translated_sites_in_enlarged_coord_system(
            old_site1)
        new_sites2 = get_all_translated_sites_in_enlarged_coord_system(
            old_site2)
        return [
            BravaisLattice.Edge(
                site2.bravais_coords - site1.bravais_coords,
                np.array([site1.subl_idx, site2.subl_idx])
            ) \
            for site1, site2 in zip(new_sites1, new_sites2)
        ]

    new_edges = []
    for old_edge in old_lattice.edges:
        new_edges += get_all_translated_edges_in_enlarged_coord_system(old_edge)

    new_hisym_points = dict(
        (k, trans_passive @ v) \
        for k, v in old_lattice.reciprocal_lattice.high_symmetry_points.items()
    )

    new_lattice = BravaisLattice(
        new_bravais_vecs,
        new_sublattices,
        new_edges,
        new_hisym_points
    )


    new_interactions = []
    for old_inter in model.interactions:
        all_translated_sites = [
            get_all_translated_sites_in_enlarged_coord_system(site) \
            for site in old_inter.sites
        ]   
        new_interactions += list(map(
            lambda sites: Interaction(
                sites, old_inter.interaction_tensor.copy()),
            zip(*all_translated_sites)
        ))

    new_classical_gs = np.tile(model.classical_gs, (scale_factor, 1))

    model_with_enlarged_unit_cell = Model(
        new_lattice,
        new_interactions,
        new_classical_gs,
    )

    return model_with_enlarged_unit_cell


"""
Reduce the dimensionality of the lattice by deleting the dimensions indexed by
the entries of dims. The embedding dimension remains the same.
The number of vectors in new_bravais_vecs should equal the old lattice dimension
minus the number of removed dimensions (i.e., the length of dims).
The dimension of each new bravais vector must equal the embedding dimension.
"""
def delete_lattice_dimensions(model: Model, dims, new_bravais_vecs,
                              new_high_symmetry_points=None,
                              new_classical_ground_state=None):

    if new_bravais_vecs.shape[0] != model.lattice.dim - len(dims):
        raise Exception(f"number supplied new bravais vectors " + 
                        f"{new_bravais_vecs.shape[0]} must equal the " +
                        f"lattice dimension of the new lower-dimensional " +
                        f"lattice {model.lattice.dim - len(dims)}")
    
    if new_bravais_vecs.shape[1] != model.lattice.embedding_dim:
        raise Exception(f"dimension of supplied new bravais vectors " + 
                        f"{new_bravais_vecs.shape[1]} must equal the " +
                        f"embedding dimension of the lattice " +
                        f"{model.lattice.embedding_dim}")

    new_edges = [
        BravaisLattice.Edge(
            np.delete(edge.bravais_coords, dims),
            edge.subl_idxs.copy()
        ) \
        for edge in model.lattice.edges \
        if np.all(np.take(edge.bravais_coords, dims, axis=0) == 0)
    ]

    if new_high_symmetry_points is None:
        new_high_symmetry_points = dict(
            (k, np.delete(v, dims)) \
            for k, v in model.lattice.reciprocal_lattice. \
                        high_symmetry_points.items()
        )

    new_lattice = BravaisLattice(
        new_bravais_vecs, model.lattice.sublattices.copy(),
        new_edges, new_high_symmetry_points
    )

    new_interactions = [
        Interaction(
            [
                BravaisLattice.Site(
                    np.delete(site.bravais_coords, dims),
                    site.subl_idx
                ) \
                for site in inter.sites
            ],
            inter.interaction_tensor.copy()
        ) \
        for inter in model.interactions
        if all(map(
            lambda site: np.all(np.take(site.bravais_coords, dims, axis=0) == 0),
            inter.sites
        ))
    ]

    if new_classical_ground_state is None:
        new_classical_ground_state = model.classical_gs.copy()

    new_model = Model(
        new_lattice,
        new_interactions,
        new_classical_ground_state
    )


    return new_model



def add_open_bc(model: Model, open_bc_config: str, slab_sizes,
                new_classical_ground_state=None):
    config = model.lattice.open_bc_configs[open_bc_config]
    return add_custom_open_bc(model, slab_sizes,
                              config["surface"], config["normal"],
                              new_classical_ground_state)

"""
add open boundary conditions along slab_sizes
"""
def add_custom_open_bc(model: Model, slab_sizes, slab_surface_coords,
                       slab_normal_coords, new_classical_ground_state=None):
    dim = model.lattice.dim
    new_dim = slab_surface_coords.shape[0]
    if dim >= 3 or dim <= 0:
        raise Exception(f"lattice dimension is {dim}, but only " \
                       + "1 and 2-dimensional lattices are supported yet.")
    
    if new_dim < 0 or new_dim > dim:
        raise Exception(f"dimension of surface {slab_surface_coords.shape[0]} "\
                      + f"must be between 0 and the lattice dimension {dim}")
    
    if slab_normal_coords.shape[0] != len(slab_sizes):
        raise Exception(f"dimension of remaining bulk " \
                      + f"{slab_surface_coords.shape[0]} must equal the " \
                      + f"number of slab dimensions {len(slab_sizes)}")
    
    if new_dim + slab_normal_coords.shape[0] != dim:
        raise Exception(f"dimension of surface {new_dim} "\
                      + f"plus dimension of remaining bulk " \
                      + f" {slab_normal_coords.shape[0]} must equal the "\
                      + f"lattice dimension {dim}")
    
    if new_dim >= 2 and slab_surface_coords.shape[1] != dim:
        raise Exception(f"length of each surface vector " \
                      + f"{slab_surface_coords.shape[1]} must equal the " \
                      + f"lattice dimension {dim}")
    
    if slab_normal_coords.shape[1] != dim:
        raise Exception(f"length of each normal vector " \
                      + f"{slab_normal_coords.shape[1]} must equal the " \
                      + f"lattice dimension {dim}")
    
    scaled_normal_vecs = np.array(slab_sizes) * slab_normal_coords
    bravais_transf = np.concatenate(
        (slab_surface_coords, scaled_normal_vecs))
    model_enlarged_unit_cell = transform(model, bravais_transf)

    if new_dim == 0:
        new_bravais_vecs = np.zeros((0, 1))
    elif new_dim == 1:
        new_bravais_vecs = np.array([
            slab_surface_coords[0] @ model.lattice.bravais_vecs,
            # [np.linalg.norm(
            #     slab_surface_coords[0] @ model.lattice.bravais_vecs
            # )]
        ])
    elif new_dim == 2:
        new_bravais_vecs = slab_surface_coords @ model.lattice.bravais_vecs
    else:
        raise Exception("lattice dimension out of range " \
                      + "(should not be reached)")
        
    strip_model = delete_lattice_dimensions(
        model_enlarged_unit_cell,
        np.arange(dim - len(slab_sizes), dim),
        new_bravais_vecs,
        new_classical_ground_state=new_classical_ground_state
    )

    return strip_model

