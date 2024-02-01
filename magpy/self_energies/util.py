

def convert_ph_labels_to_indices(particle_hole_labels):
    def map_label_to_idx(ph):
        if ph == "p":
            return 1
        elif ph == "h":
            return 0
        else:
            raise Exception(f"invalid particle-hole state {ph}: "
                          + f"m be either p or h.")
        
    particle_hole_idxs = []
    for ph_label in particle_hole_labels:
        particle_hole_idxs.append(list(map(
            map_label_to_idx, ph_label
        )))

    return particle_hole_idxs



def to_binary(bits):
    return sum(2**i * bit for i, bit in enumerate(reversed(bits)))