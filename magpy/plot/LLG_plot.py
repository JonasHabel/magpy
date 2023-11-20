import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    
    
    def animate(step, qr):
        qr.set_UVC(spin_configs[:,np.newaxis, 0, step % num_times],
                   spin_configs[:,np.newaxis, 1, step % num_times],
                   0.5*(spin_configs[:, np.newaxis, 2, step % num_times] + 1))
        return qr
        
    qr = ax.quiver(lattice_sites_pos[:, np.newaxis, 0],
                   lattice_sites_pos[:, np.newaxis, 1],
                   spin_configs[:, np.newaxis, 0, 0],
                   spin_configs[:, np.newaxis, 1, 0],
                   0.5*(spin_configs[:, np.newaxis, 2, 0] + 1), **params)
    
    anim = FuncAnimation(fig, animate, frames=num_times, fargs=(qr,),
                         interval=100, blit=False)
    
    return fig, ax, anim