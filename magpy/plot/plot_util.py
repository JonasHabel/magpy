import numpy as np
import matplotlib.pyplot as plt


def set_momentum_path_x_ticks(ax, momentum_path, krange=None, high_sym_point_label_map={},
                              vertical_lines_params=None):
    if krange is None:
        krange = 0, len(momentum_path.ks)
    
    ax.set_xticks([
        k_idx for k_idx in momentum_path.high_sym_point_idxs \
            if krange[0] <= k_idx < krange[1]
    ])
    high_sym_point_labels = [
        high_sym_point_label_map.get(label, label) \
        for k_idx, label in zip(
            momentum_path.high_sym_point_idxs,
            momentum_path.high_sym_point_labels) \
        if krange[0] <= k_idx < krange[1]
    ]
    ax.set_xticklabels(high_sym_point_labels)

    if vertical_lines_params:
        set_momentum_path_vertical_lines(ax, momentum_path,
                                         params=vertical_lines_params)

def set_momentum_path_vertical_lines(ax, momentum_path, krange=None, params={}):
    if krange is None:
        krange = 0, len(momentum_path.ks)

    for k_idx in momentum_path.high_sym_point_idxs:
        if krange[0] <= k_idx < krange[1]:
            ax.vlines(k_idx, *ax.get_ylim(), **params)



def map_spin_component_to_color(spin_config, S_max, cmap):
    return cmap((spin_config + S_max) / (2*S_max))


def quiver(ax, lattice_sites_pos, spin_config, S_max, cmap, params):
    has_cmap = cmap is not None
    if lattice_sites_pos.shape[-1] == 1:    # embedding dimension is 1
        lattice_sites_pos = np.hstack((lattice_sites_pos, np.zeros((lattice_sites_pos.shape[0], 1), dtype=lattice_sites_pos.dtype)))

    if has_cmap:
        cmap = plt.get_cmap(cmap)
        colors = map_spin_component_to_color(
            spin_config[:, np.newaxis, 2], S_max, cmap)
        return ax.quiver(lattice_sites_pos[:, np.newaxis, 0],
                         lattice_sites_pos[:, np.newaxis, 1],
                         spin_config[:, np.newaxis, 0],
                         spin_config[:, np.newaxis, 1],
                         color=colors, 
                         **{k: v for k, v in params.items() if k != "cmap"})
    else:
        return ax.quiver(lattice_sites_pos[:, np.newaxis, 0],
                         lattice_sites_pos[:, np.newaxis, 1],
                         spin_config[:, np.newaxis, 0],
                         spin_config[:, np.newaxis, 1],
                         spin_config[:, np.newaxis, 2], 
                         **params)


def update_quiver(qr, spin_config, S_max, cmap):
    has_cmap = cmap is not None
    qr.set_UVC(spin_config[:, np.newaxis, 0],
               spin_config[:, np.newaxis, 1],
               spin_config[:, np.newaxis, 2] if not has_cmap else None)
    if has_cmap:
        colors = map_spin_component_to_color(
            spin_config[:, np.newaxis, 2], S_max, cmap)
        qr.set_color(colors)
