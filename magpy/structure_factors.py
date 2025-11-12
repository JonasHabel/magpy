import numpy as np
from scipy.linalg import block_diag
from magpy.models import Model


"""
magnon_greens_functions = [
    [<a a^†>], [<a a>],
    [<a^† a^†>], [<a^† a],
]

"""
def compute_structure_factor(magnon_greens_functions, momentum, eigvs, model: Model):
    assert len(magnon_greens_functions.shape) == 3
    assert len(eigvs.shape) == 2
    assert magnon_greens_functions.shape[1] % 2 == 0

    num_freqs = magnon_greens_functions.shape[0]
    num_bands = magnon_greens_functions.shape[1] // 2

    assert magnon_greens_functions.shape[2] == 2*num_bands
    assert eigvs.shape[0] == eigvs.shape[1] == 2*num_bands

    eigvs_conj = eigvs.T.conj()
    Hadamard = np.kron(np.eye(num_bands), np.array([[1, 1], [-1j, 1j]]))
    gs_rot_mat = model.compute_ground_state_rotation_matrices()[:, :, 0:2]
    gs_rot_mat_block = block_diag(*gs_rot_mat)

    # we need to transpose here because we accidentally chose a suboptimal
    # convention for the indices of the Green's function: G[..., in, out]
    # where the "standard" convention is G[..., out, in]
    GF_eigenspace_basis = magnon_greens_functions.transpose((0, 2, 1))
    GF_sublattice_basis = \
        np.einsum("sn,...nm,mt->...st", eigvs, GF_eigenspace_basis, eigvs_conj)
    GF_spin_basis_rotated_frame = \
        np.einsum("as,...st,tb->...ab", Hadamard, GF_sublattice_basis, Hadamard.T.conj())
    GF_spin_basis_lab_frame = \
        np.einsum("Aa,...ab,bB->...AB", gs_rot_mat_block, GF_spin_basis_rotated_frame, gs_rot_mat_block.T)

    S = model.get_onsite_spin_quantum_numbers()
    sublattice_offsets = model.lattice.sublattices
    u = np.sqrt(S) * np.exp(1j*sublattice_offsets.dot(momentum)) # the sublattice structure factor
    u_matrix = np.kron(np.outer(u.conj(), u), np.eye(3))

    # S_{αβ}(ω)   (α,β = x,y,z)
    structure_factor = np.zeros((num_freqs, 3, 3), dtype=np.complex128)

    for n, GF_for_freq in enumerate(GF_spin_basis_lab_frame):
        # the "missing" minus sign in front of 0.5j is correct because our Greens function is
        # actually defined as -1/(w - E - Sigma)
        GF_antiherm = 0.5j * (GF_for_freq - GF_for_freq.T.conj())

        # reshaped axes: (sublattice_i, spin-dir_i, sublattice_j, spin-dir_j)
        # where spin-dir_i and spin-dir_j are in the lab frame (x, y, z)
        struct_fact_sublattice = \
            (u_matrix @ GF_antiherm).reshape((num_bands, 3, num_bands, 3))
        structure_factor[n] = np.sum(
            struct_fact_sublattice,
            axis=(0, 2),
        )   # sum over sublattices

    return structure_factor


def apply_kinetic_projector(structure_factor, momentum, model):
    gs_rot_mat = model.compute_ground_state_rotation_matrices()[0]  
    structure_factor_lab_frame = np.einsum("...AB,Aa,Bb->ab", structure_factor, )
    kinetic_projector = np.eye(2) - np.outer(momentum[0:2], momentum[0:2]) / momentum.dot(momentum)

    return np.einsum("...ab,ab", structure_factor, kinetic_projector)


def apply_kinetic_projectors(structure_factors, momenta):
    projected_str_fac = np.zeros(structure_factors.shape[:-2], dtype=np.float64)
    for n, (str_fac, k) in enumerate(zip(structure_factors, momenta)):
        projected_str_fac[n] = apply_kinetic_projector(str_fac, k)
    
    return projected_str_fac
