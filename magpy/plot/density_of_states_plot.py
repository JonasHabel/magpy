import matplotlib.pyplot as plt
import numpy as np
from ..lattice import ReciprocalLattice
from .plot_util import set_momentum_path_x_ticks


def plot_n_magnon_density_of_states_along_momentum_path(
        momentum_path: ReciprocalLattice.MomentumPath,
        frequencies, n_magnon_dos, params={}):
    fig, ax = plt.subplots()
    plot_ax_n_magnon_density_of_states_along_momentum_path(
        momentum_path, frequencies, n_magnon_dos, params, fig, ax)
    set_momentum_path_x_ticks(ax, momentum_path)
    plt.show()


def plot_ax_n_magnon_density_of_states_along_momentum_path(
        momentum_path: ReciprocalLattice.MomentumPath,
        frequencies, n_magnon_dos, params={}, fig=plt.gcf(), ax=plt.gca()):
    c = ax.contourf(np.arange(len(momentum_path.ks)),
                frequencies, n_magnon_dos.T, **params)
    cbar = fig.colorbar(c)
    
    return fig, ax, cbar