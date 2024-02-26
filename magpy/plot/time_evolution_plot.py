from matplotlib.animation import FuncAnimation
import numpy as np
from magpy.lattice import BravaisLattice
import matplotlib.pyplot as plt

def plot_time_evolution(wavefunctions, times, lattice: BravaisLattice, 
        params={}, anim_params={}):
    fig, ax = plt.subplots()
    _, _, anim = plot_ax_time_evolution(
        wavefunctions, times, lattice, 
        params, anim_params, fig, ax)
    plt.show()
    return anim


def plot_ax_time_evolution(
        wavefunctions, times, lattice: BravaisLattice, 
        params={}, anim_params={}, fig=plt.gcf(), ax=plt.gca()):
    assert lattice.embedding_dim <= 2

    lattice_dims = wavefunctions.shape[1:-1]
    num_unit_cells = int(np.prod(lattice_dims))
    num_sublattices = lattice.num_sites_unit_cell
    num_sites_total = num_unit_cells * num_sublattices
    num_times = wavefunctions.shape[0]
    sites_pos = lattice.sample_full_lattice_in_canonical_coords(lattice_dims)

    wavefunctions_flat = wavefunctions.reshape((num_times, num_sites_total))
    prob_densities_flat = np.abs(wavefunctions_flat)**2
    max_prob_density = np.amax(prob_densities_flat)
    max_blob_size = params.get("max_blob_size", 200)
    blob_sizes_flat = max_blob_size * prob_densities_flat / max_prob_density
    blob_colors = (np.angle(wavefunctions_flat) / (2*np.pi)) % 1.0
    
    sites_pos_flat = sites_pos.reshape((num_sites_total, lattice.embedding_dim))
    padded_sites_pos_flat = (
        *sites_pos_flat.T, 
        *np.zeros((2 - lattice.embedding_dim, num_sites_total))
    )

    def animate(animation_step, scat):
        scat.set_sizes(blob_sizes_flat[animation_step])
        scat.set_array(blob_colors[animation_step])
        return scat

    if "cmap" not in params:
        params["cmap"] = "hsv"
    scat = ax.scatter(*padded_sites_pos_flat, s=blob_sizes_flat[0], c=blob_colors[0], **params)

    anim = FuncAnimation(fig, animate, frames=num_times, fargs=(scat,),
                         **anim_params)
    
    return fig, ax, anim