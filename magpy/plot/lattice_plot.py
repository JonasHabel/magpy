import matplotlib.pyplot as plt
import numpy as np
from ..lattice import *

def plot_lattice(lattice: BravaisLattice, sizes):
    fig = plt.figure()
    if lattice.dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    __plot_lattice_internal(ax, lattice, sizes)
    plt.show()


def __plot_lattice_internal(ax, lattice: BravaisLattice, sizes):
    dim = lattice.dim
    if dim >= 4:
        raise Exception("plot_lattice for lattice dimensions >= 4 not " \
                      + "implemented yet")
    if dim == 0:
        __plot_unit_cell_internal(ax, np.zeros(0), lattice, None)
        return
    
    sizes = np.array(sizes)
    grid = np.array(np.meshgrid(*[
        np.arange(size) for size in sizes
    ])).reshape((dim, np.prod(sizes))).T

    for unit_cell_bravais_coords in grid:
        __plot_unit_cell_internal(ax, unit_cell_bravais_coords, lattice, sizes)



def plot_unit_cell(lattice: BravaisLattice):
    fig, ax = plt.subplots()
    __plot_unit_cell_internal(ax, np.zeros(lattice.dim),
                              lattice, sizes=np.ones(lattice.dim))
    plt.show()


def __plot_unit_cell_internal(ax, unit_cell_bravais_coords,
                              lattice: BravaisLattice, sizes=None):
    unit_cell_bravais_pos = lattice.bravais_vecs.T @ unit_cell_bravais_coords
    MAX_SUPPORTED_DIM = 3
    for edge in lattice.edges:
        # truncate at the boundaries to avoid dangling edges
        if sizes is not None and \
           (np.any(unit_cell_bravais_coords + edge.bravais_coords >= sizes) or \
            np.any(unit_cell_bravais_coords + edge.bravais_coords < 0)):
            continue
        positions = np.array([
            lattice.sublattices[int(subl_idx), 0:MAX_SUPPORTED_DIM] \
            for subl_idx in edge.subl_idxs
        ]).astype(float)
        if lattice.dim >= 1:
            edge_bravais_vec = lattice.bravais_vecs.T @ edge.bravais_coords
            positions[1, 0:len(edge_bravais_vec)] \
                += edge_bravais_vec
            positions[:, 0:len(unit_cell_bravais_pos)] \
                += unit_cell_bravais_pos

        linestyle = "solid" \
            if np.all(edge.bravais_coords == 0) \
            else "dotted"
        plot_params = dict(color="black", linewidth=2, linestyle=linestyle)
        if positions.shape[1] == 1:
            ax.plot(positions[:,0], np.zeros(len(positions)), **plot_params)
        elif positions.shape[1] == 2:
            ax.plot(*(positions.T), **plot_params)
        elif positions.shape[1] == 3:
            ax.plot3D(*(positions.T), **plot_params)
        else:
            raise Exception()
        