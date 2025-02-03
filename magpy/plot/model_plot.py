import matplotlib.pyplot as plt
import numpy as np
from ..lattice import *
from ..models import Model
from ..interactions import Interaction
from . import lattice_plot
from ..util import LEVI_CIVITA

def plot_model(model: Model, sizes):
    fig = plt.figure()
    if model.lattice.dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    plot_ax_model(model, sizes, ax)
    plt.show()


def plot_ax_model(model: Model, sizes, ax):
    dim = model.lattice.dim
    if dim >= 4:
        raise Exception("plot_lattice for lattice dimensions >= 4 not " \
                      + "implemented yet")
    if dim == 0:
        plot_ax_unit_cell(ax, np.zeros(0), model, None)
        return
    
    sizes = np.array(sizes)
    grid = model.lattice.sample_Bravais_lattice_in_Bravais_coords(sizes)
    interactions_by_sites = model.group_interactions_by_sites()

    for unit_cell_bravais_coords in grid:
        plot_ax_unit_cell_internal(
            unit_cell_bravais_coords, model, ax, sizes, interactions_by_sites)

    return ax



def plot_unit_cell(model: Model):
    fig, ax = plt.subplots()
    plot_ax_unit_cell(np.zeros(model.lattice.dim),
                      model, ax, sizes=np.ones(model.lattice.dim))
    plt.show()


def plot_ax_unit_cell(unit_cell_bravais_coords, 
                      model: Model, ax, sizes=None):
    return plot_ax_unit_cell_internal(
        unit_cell_bravais_coords, model, ax, sizes)


def plot_ax_unit_cell_internal(unit_cell_bravais_coords, model: Model,
                               ax, sizes=None, interactions_by_sites=None):
    interactions_by_sites = interactions_by_sites \
        if interactions_by_sites is not None \
        else model.group_interactions_by_sites()
    
    for interaction in interactions_by_sites:
        plot_ax_interaction(
            interaction, unit_cell_bravais_coords, model, ax, sizes)



def plot_ax_interaction(interaction: Interaction, unit_cell_bravais_coords,
                        model: Model, ax, sizes=None):
    WHITE = np.array([1.0, 1.0, 1.0])
    HEISENBERG_SATURATION_COLOR = np.array([1.0, 0.4, 0.0])   # orange
    HEISENBERG_SATURATION_VALUE = 1.0
    DMI_SATURATION_COLOR = np.array([1.0, 1.0, 0.0])   # yellow
    DMI_SATURATION_VALUE = 1.0

    def to_color(value, saturation_value, saturation_color):
        value = abs(value)
        saturation_value = abs(saturation_value)
        clipped_value = min(value, saturation_value)
        ratio = clipped_value / saturation_value
        return tuple(saturation_color*ratio + WHITE*(1-ratio))

    if len(interaction.sites) != 2:
        # only ploting of two-site interactions supported so far
        return


    edge = BravaisLattice.Edge.from_sites(*interaction.sites)
    int_tensor = interaction.interaction_tensor
    Heisenberg_contribution = np.trace(int_tensor) / 3
    DMI_contribution = 0.5 * np.linalg.norm(np.einsum("ijk,ij", LEVI_CIVITA, 0.5*(int_tensor - int_tensor.T)))

    if abs(Heisenberg_contribution) > 1e-10:
        lattice_plot.plot_ax_edge(
            edge,
            unit_cell_bravais_coords + interaction.sites[0].bravais_coords,
            model.lattice, ax, sizes, 
            lambda _: dict(color=to_color(
                Heisenberg_contribution, HEISENBERG_SATURATION_VALUE, HEISENBERG_SATURATION_COLOR,
            )))
    
    if abs(DMI_contribution) > 1e-10:
        lattice_plot.plot_ax_edge(
            edge,
            unit_cell_bravais_coords + interaction.sites[0].bravais_coords,
            model.lattice, ax, sizes, 
            lambda _: dict(color=to_color(
                DMI_contribution, DMI_SATURATION_VALUE, DMI_SATURATION_COLOR,
            )))