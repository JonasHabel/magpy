import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.quiver import Quiver
from magpy.plot import plot_util
import numpy as np

def plot_LLG_animation(times, spin_configs, lattice, params={}):
    fig, ax = plt.subplots()
    _, _, anim = plot_ax_LLG_animation(times, spin_configs, lattice, params, fig, ax)
    plt.show()
    return anim


def plot_ax_LLG_animation(times, spin_configs, lattice, params={},
                          fig=plt.gcf(), ax=plt.gca()):
    if lattice.dim >= 3:
        raise Exception("So far, only <3-dimensional lattices are supported.")
    
    sizes = np.array(spin_configs.shape[:-3], dtype=int)
    num_unit_cells = np.prod(sizes)
    num_sites_total = num_unit_cells * lattice.num_sites_unit_cell
    lattice_sites_pos = lattice \
        .sample_full_lattice_in_canonical_coords(sizes) \
        .reshape((num_sites_total, lattice.sublattices.shape[-1]))
    num_times = len(times)
    spin_configs = spin_configs.reshape(
        (num_sites_total, 3, num_times))
    # normalize wrt largest spin quantum number
    S_max = np.amax(np.linalg.norm(spin_configs[:, :, 0], axis=1))
    has_cmap = "cmap" in params
    cmap = plt.get_cmap(params["cmap"]) if has_cmap else None
    
    def animate(step, qr: Quiver):
        plot_util.update_quiver(
            qr, spin_configs[:, :, step % num_times], S_max, cmap)
        return qr
        
    qr = plot_util.quiver(
        ax, lattice_sites_pos, spin_configs[:, :, 0], S_max, cmap, params)
    
    anim = FuncAnimation(fig, animate, frames=num_times, fargs=(qr,),
                         interval=100, blit=False)
    
    return fig, ax, anim