
def set_momentum_path_x_ticks(ax, momentum_path, high_sym_point_label_map={},
                              vertical_lines_params=None):
    ax.set_xticks([
        k_idx for k_idx in momentum_path.high_sym_point_idxs \
            if 0 <= k_idx < len(momentum_path.ks)
    ])
    high_sym_point_labels = [
        high_sym_point_label_map.get(label, label) \
        for k_idx, label in zip(
            momentum_path.high_sym_point_idxs,
            momentum_path.high_sym_point_labels) \
        if 0 <= k_idx < len(momentum_path.ks)
    ]
    ax.set_xticklabels(high_sym_point_labels)

    if vertical_lines_params:
        set_momentum_path_vertical_lines(ax, momentum_path,
                                         params=vertical_lines_params)

def set_momentum_path_vertical_lines(ax, momentum_path, params={}):
    for k_idx in momentum_path.high_sym_point_idxs:
        if 0 <= k_idx < len(momentum_path.ks):
            ax.vlines(k_idx, *ax.get_ylim(), **params)