import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from magpy.classical.util import convert_to_flat_index
from magpy.plot import plot_util

def plot_monte_carlo_animation(
        update_infos, step_size, init_spin_config, lattice,
        params={}, anim_params={}):
    fig, ax = plt.subplots()
    _, _, anim = plot_ax_monte_carlo_animation(
        update_infos, step_size, init_spin_config, lattice, 
        params, anim_params, fig, ax
    )
    plt.show()
    return anim


def plot_ax_monte_carlo_animation(
        update_infos, step_size, init_spin_config, lattice, 
        params={}, anim_params={}, fig=plt.gcf(), ax=plt.gca()):
    if lattice.dim >= 3:
        raise Exception("So far, only <3-dimensional lattices are supported.")
    
    lattice_sizes = np.array(init_spin_config.shape[:-2], dtype=int)
    num_unit_cells = np.prod(lattice_sizes)
    num_sites_total = num_unit_cells * lattice.num_sites_unit_cell
    lattice_sites_pos = lattice \
        .sample_full_lattice_in_canonical_coords(lattice_sizes) \
        .reshape((num_sites_total, lattice.sublattices.shape[-1]))
    accept, bravais_coords, subl_idxs, spins = update_infos
    num_steps = len(accept) // step_size
    init_spin_config = init_spin_config.reshape((num_sites_total, 3))
    spin_config = init_spin_config.copy() # avoid side effects

    # normalize wrt largest spin quantum number
    S_max = np.amax(np.linalg.norm(spin_config, axis=1))
    has_cmap = "cmap" in params
    cmap = plt.get_cmap(params["cmap"]) if has_cmap else None
    
    def animate(animation_step, qr):
        for n in range(step_size):
            intermediate_step_idx = animation_step * step_size + n
            if not accept[intermediate_step_idx]:
                continue
            flat_idx = convert_to_flat_index(
                bravais_coords[intermediate_step_idx], 
                subl_idxs[intermediate_step_idx], 
                lattice_sizes, lattice.num_sites_unit_cell
            )
            spin_config[flat_idx] = spins[intermediate_step_idx]
            
        plot_util.update_quiver(qr, spin_config, S_max, cmap)

        return qr
        
    qr = plot_util.quiver(
        ax, lattice_sites_pos, spin_config, S_max, cmap, params)
    
    anim = FuncAnimation(fig, animate, frames=num_steps, fargs=(qr,),
                         **anim_params)
    
    return fig, ax, anim