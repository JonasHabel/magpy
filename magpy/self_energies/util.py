import numpy as np
from ..util import PAULI_MATRICES


def convert_ph_labels_to_indices(particle_hole_labels):
    def map_label_to_idx(ph):
        if ph == "p":
            return 1
        elif ph == "h":
            return 0
        else:
            raise Exception(f"invalid particle-hole state {ph}: "
                          + f"must be either p or h.")
        
    particle_hole_idxs = []
    for ph_label in particle_hole_labels:
        particle_hole_idxs.append(list(map(
            map_label_to_idx, ph_label
        )))

    return particle_hole_idxs



def to_binary(bits):
    return sum(2**i * bit for i, bit in enumerate(reversed(bits)))



def conjugate_if(eigvs, condition):
    num_bands = eigvs.shape[-1] // 2
    assert 2*num_bands == eigvs.shape[-2]
    identity = np.eye(num_bands)
    sigma_x = np.kron(identity, PAULI_MATRICES[0])

    return sigma_x @ eigvs.conj() @ sigma_x if condition() else eigvs




def compute_gauge_phase(eigvs_1, eigvs_2):
    num_bands = eigvs_1.shape[-1] // 2
    assert 2*num_bands == eigvs_1.shape[-2]
    assert 2*num_bands == eigvs_2.shape[-1]
    assert 2*num_bands == eigvs_2.shape[-2]

    identity = np.eye(num_bands)
    sigma_x = np.kron(identity, PAULI_MATRICES[0])
    sigma_z = np.kron(identity, PAULI_MATRICES[2])
    eigvs_1_inv = sigma_z @ eigvs_1.T.conj() @ sigma_z

    return np.diag(eigvs_1_inv @ sigma_x @ eigvs_2.conj() @ sigma_x)