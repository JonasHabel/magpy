import numpy as np
from .lattice import BravaisLattice, ReciprocalLattice, DotLattice
from . import lattice
from .interactions import Interaction, CompositeInteraction, compute_rotated_interactions, compress
from . import interactions
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
                 classical_ground_state, interaction_compression=None):
        classical_gs_required_shape = (lattice.num_sites_unit_cell, 3)
        self.classical_gs = np.array(classical_ground_state)
        if self.classical_gs.shape != classical_gs_required_shape:
            raise Exception("classical ground state array has wrong shape " \
                + f"{classical_ground_state.shape}. Should be " \
                + f"{classical_gs_required_shape}")
        
        self.lattice = lattice
        self.interactions = CompositeInteraction(interactions).expand()

        if interaction_compression is not None:
            self.interactions = compress(
                self.interactions, **interaction_compression
            )

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
        return compute_rotated_interactions(
            self.interactions, self.compute_ground_state_rotation_matrices()
        )

    """
    Group interactions acting on the same sites into a single interaction
    and convert them into a dict {tuple(sites) -> Interaction}
    """
    def compress_interactions(self):
        return compress(self.interactions)
    

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

    new_lattice = lattice.stack(model.lattice, num_layers,
          interlayer_edges,
          additional_high_symmetry_points,
          distance_between_layers,
          sublattice_shifts,
          periodic,
          additional_bravais_vec)

    old_num_sites_unit_cell = model.lattice.num_sites_unit_cell

    new_interactions = []
    for inter in model.interactions:
        new_inters = [
            Interaction([
                BravaisLattice.Site(
                    np.r_[site.bravais_coords, 0] \
                        if periodic else site.bravais_coords,
                    site.subl_idx + old_num_sites_unit_cell*layer
                ) for site in inter.sites
            ], inter.interaction_tensor.copy()) \
            for layer in range(num_layers)
        ]
        new_interactions += new_inters
    new_interactions += interlayer_interactions

    if new_classical_ground_state is None:
        new_classical_ground_state = np.tile(
            model.classical_gs, (num_layers, 1)
        )

    stacked_model = Model(
        new_lattice, new_interactions, new_classical_ground_state
    )


    return stacked_model


"""
Form the "direct product" of model1 and model2.
model1 and model2 must have the same underlying Bravais lattice.
inter_model_interactions specifies additional "off-diagonal" interactions
between sites of model1 and sites of model2.
"""
def direct_product(model1: Model, model2: Model,
        inter_model_edges=[],
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

    new_edges = old_lattice1.edges \
        + list(map(
            lambda edge: BravaisLattice.Edge(
                edge.bravais_coords.copy(),
                edge.subl_idxs + old_lattice1.num_sites_unit_cell
            ), old_lattice2.edges)) \
        + inter_model_edges

    new_lattice = BravaisLattice(
        old_lattice1.bravais_vecs.copy(),
        new_sublattices,
        new_edges,
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
def transform(model: Model, bravais_trans, new_classical_ground_state=None):
    dim = bravais_trans.shape
    if len(dim) != 2 or dim[0] != model.lattice.dim or dim[0] != dim[1]:
        raise Exception(f"lattice dimension {model.lattice.dim} " +
                        f" and bravais_trans dimensions {dim} should be " +
                        f"the same.")

    old_lattice = model.lattice
    old_num_sites_unit_cell = old_lattice.num_sites_unit_cell

    new_lattice, transform_utils = lattice.transform(
        model.lattice, bravais_trans, __return_with_transform_utils=True)

    new_interactions = []
    for old_inter in model.interactions:
        all_translated_sites = [
            transform_utils.get_all_translated_sites_in_enlarged_coord_system(
                site, transform_utils.new_unit_cell_sites_coords
            ) for site in old_inter.sites
        ]   
        new_interactions += list(map(
            lambda sites: Interaction(
                sites, old_inter.interaction_tensor.copy()),
            zip(*all_translated_sites)
        ))

    trans_active = transform_utils.trans_active
    scale_factor = int(np.round(np.abs(np.linalg.det(trans_active))))
    new_num_sites_unit_cell = old_num_sites_unit_cell * scale_factor
    if new_classical_ground_state is None:
        new_classical_gs = np.tile(model.classical_gs, (scale_factor, 1))
    elif new_classical_ground_state.shape != (new_num_sites_unit_cell, 3):
        raise Exception("new classical ground state array has wrong shape " \
                + f"{new_classical_ground_state.shape}. Should be " \
                + f"{(new_num_sites_unit_cell, 3)}")
    else:
        new_classical_gs = new_classical_ground_state

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
def delete_dimensions(model: Model, dims, new_bravais_vecs,
                      new_high_symmetry_points=None,
                      new_classical_ground_state=None,
                      periodic_boundary_conditions=False):

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
    
    new_lattice = lattice.delete_dimensions(
        model.lattice, dims, new_bravais_vecs,
        new_high_symmetry_points,
        periodic_boundary_conditions
    )

    new_interactions = [
        Interaction(
            [
                BravaisLattice.Site(
                    filtered_bravais_coords_for_site,
                    site.subl_idx
                ) \
                for site, filtered_bravais_coords_for_site in zip(
                    inter.sites, filtered_bravais_coords
                )
            ],
            inter.interaction_tensor.copy()
        ) \
        for inter, filtered_bravais_coords in lattice.__DeleteDimensionsUtils.filter(
            model.interactions,
            lambda inter: np.array([site.bravais_coords for site in inter.sites]),
            dims, periodic_boundary_conditions,
        )
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
    return add_custom_bc(
        model, slab_sizes, slab_surface_coords, slab_normal_coords,
        periodic=False,
        new_classical_ground_state=new_classical_ground_state
    )
    


def add_custom_bc(model: Model, slab_sizes, slab_surface_coords,
        slab_normal_coords, periodic, new_classical_ground_state=None):
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
        new_bravais_vecs = np.zeros((0, model.lattice.embedding_dim))
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
        
    strip_model = delete_dimensions(
        model_enlarged_unit_cell,
        np.arange(dim - len(slab_sizes), dim),
        new_bravais_vecs,
        new_classical_ground_state=new_classical_ground_state,
        periodic_boundary_conditions=periodic,
    )

    return strip_model





def rearrange_sublattices(model: Model, permutation):
    new_lattice = lattice.rearrange_sublattices(model.lattice, permutation)

    new_interactions = [
        Interaction(
            sites=[
                BravaisLattice.Site(
                    bravais_coords=site.bravais_coords,
                    sublattice_index=permutation[site.subl_idx]
                ) for site in inter.sites
            ],
            interaction_tensor=inter.interaction_tensor,
        ) for inter in model.interactions
    ]

    new_classical_ground_state = util.permute(model.classical_gs, permutation)

    new_model = Model(new_lattice, new_interactions, new_classical_ground_state)

    return new_model




def remove_sublattices(model: Model, subl_idxs):
    new_lattice, utils = lattice.remove_sublattices(
        model.lattice, subl_idxs, __return_with_utils=True,
    )

    new_interactions = [
        Interaction(
            sites=[
                BravaisLattice.Site(
                    bravais_coords=site.bravais_coords,
                    sublattice_index=utils.map_subl_idx(site.subl_idx),
                ) for site in inter.sites
            ],
            interaction_tensor=inter.interaction_tensor,
        ) for inter in model.interactions
        if all(site.subl_idx not in subl_idxs for site in inter.sites)
    ]

    new_classical_ground_state = model.classical_gs[utils.remaining_subl_idxs]

    new_model = Model(new_lattice, new_interactions, new_classical_ground_state)

    return new_model

   

