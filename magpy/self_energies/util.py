import numpy as np
from numba import njit
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

    # this should usually be a diagonal matrix;
    # except if there are degenerate eigenspaces, in which case the matrix is
    # block-diagonal. But there should be no off-diagonal terms between
    # eigenspaces pertaining to different eigenvalues.
    return eigvs_1_inv @ sigma_x @ eigvs_2.conj() @ sigma_x




def compute_diagram_prefactor(ph_idxs_verts, ph_idxs_loops, num_internal_propagators):
    order = len(ph_idxs_verts)

    return compute_diagram_sign(order, num_internal_propagators) \
         * compute_num_Wick_contractions(ph_idxs_verts, ph_idxs_loops)


@njit
def compute_diagram_sign(order, num_internal_propagators):
    return -1 * (-1)**order * (-1)**num_internal_propagators


def compute_num_Wick_contractions(ph_idxs_verts, ph_idxs_loops):
    PARTICLE, HOLE = 1, 0
    num_Wick_contractions = 1

    # possible ways to wire the vertices
    for ph_idxs_vert in ph_idxs_verts:
        ph_idxs_vert = np.array(ph_idxs_vert)
        num_particle_idxs = np.count_nonzero(ph_idxs_vert == PARTICLE)
        num_hole_idxs = np.count_nonzero(ph_idxs_vert == HOLE)
        num_Wick_contractions *= \
            np.math.factorial(num_particle_idxs) * np.math.factorial(num_hole_idxs)
        
    # account for overcounting due to loops
    for ph_idxs_loop_state in ph_idxs_loops:
        ph_idxs_loop_state = np.array(ph_idxs_loop_state)
        num_particle_idxs = np.count_nonzero(ph_idxs_loop_state == PARTICLE)
        num_hole_idxs = np.count_nonzero(ph_idxs_loop_state == HOLE)
        num_Wick_contractions /= \
            np.math.factorial(num_particle_idxs) * np.math.factorial(num_hole_idxs)
        
    assert np.allclose(num_Wick_contractions, int(num_Wick_contractions))
    return int(num_Wick_contractions)
