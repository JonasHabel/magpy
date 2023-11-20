import matplotlib.pyplot as plt
import numpy as np
from ..lattice import BravaisLattice, ReciprocalLattice
from .plot_util import set_momentum_path_x_ticks

def plot_energies_along_momentum_path(
        momentum_path: ReciprocalLattice.MomentumPath, energies, params={}):
    fig, ax = plt.subplots()
    plot_ax_energies_along_momentum_path(
        momentum_path, energies, params, fig, ax)
    set_momentum_path_x_ticks(ax, momentum_path)
    plt.show()


def plot_ax_energies_along_momentum_path(
        momentum_path: ReciprocalLattice.MomentumPath, energies, params={},
        fig=plt.gcf(), ax=plt.gca()):
    ax.plot(np.arange(len(momentum_path.momenta)), energies, **params)

    return fig, ax


"""
momenta: (2, #, #) numpy array
    meshgrid of sampled momenta
"""
def plot_energies_3D(momenta, energies, band_idxs=None, params={}):
    fig = plt.gcf()
    ax = fig.add_subplot(projection="3d")
    plot_ax_energies_3D(momenta, energies, band_idxs, fig, ax)
    plt.show()


def plot_ax_energies_3D(momenta, energies, band_idxs=None, params={},
                        fig=plt.gcf(), ax=plt.gca()):
    if momenta.shape[0] != 2:
        raise Exception(f"momenta must be 2-dim. vectors, not " \
                       + "{momenta.shape[0]}-dim. vectors.")
    
    if band_idxs is None:
        band_idxs = range(energies.shape[-1])

    for band_idx in band_idxs:
        if not np.all(energies[:, :, band_idx] < 0):
            ax.plot_surface(*momenta, energies[:, :, band_idx], **params)
    ax.set_zlim(0.0)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel(r"$\epsilon$")

    return fig, ax